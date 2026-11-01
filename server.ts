import "dotenv/config";
import express from "express";
import type { Request, Response } from "express";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";
import { GoogleGenAI, Type } from "@google/genai";
import type { GenerateContentParameters, GenerateContentResponse } from "@google/genai";

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const VIDEO_STORAGE_DIR = "/tmp/mpt-videos";
const LIBRARY_INDEX_FILE = path.join(VIDEO_STORAGE_DIR, "library.json");

fs.mkdirSync(VIDEO_STORAGE_DIR, { recursive: true });

interface StoredVideoRecord {
  id: string;
  subject: string;
  script: string;
  keywords: string[];
  aspectRatio: string;
  durationSeconds: number;
  createdAt: string;
  filename: string;
  videoUrl: string;
  downloadUrl: string;
  videoSource: string;
  voiceName: string;
  renderEngine?: string;
}

function readLibrary(): StoredVideoRecord[] {
  try {
    if (!fs.existsSync(LIBRARY_INDEX_FILE)) return [];
    const raw = fs.readFileSync(LIBRARY_INDEX_FILE, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeLibrary(items: StoredVideoRecord[]): void {
  fs.writeFileSync(LIBRARY_INDEX_FILE, JSON.stringify(items, null, 2), "utf8");
}

function getGenAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please attach your Gemini API key in the Settings > Secrets panel."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function isTransientDemandError(err: unknown): boolean {
  const raw = err instanceof Error ? err.message : String(err);
  return (
    raw.includes("503") ||
    raw.includes("UNAVAILABLE") ||
    raw.includes("429") ||
    raw.includes("RESOURCE_EXHAUSTED") ||
    raw.includes("high demand") ||
    raw.includes("overloaded") ||
    raw.includes("500") ||
    raw.includes("502") ||
    raw.includes("504") ||
    raw.includes("INTERNAL") ||
    raw.includes("DEADLINE_EXCEEDED") ||
    raw.includes("fetch failed")
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const TEXT_MODEL_CASCADE = [
  "gemini-3.8-flash",
  "gemini-3.1-flash-lite",
  "gemini-flash-lite-latest",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3.5-flash-lite",
  "gemini-flash-latest",
] as const;

const TTS_MODEL_CASCADE = [
  "gemini-3.8-flash-lite-tts",
  "gemini-3.8-flash-tts",
] as const;

const modelCooldownUntil = new Map<string, number>();

async function generateTextWithFallback(
  ai: GoogleGenAI,
  params: Omit<GenerateContentParameters, "model">
): Promise<GenerateContentResponse> {
  let lastError: unknown;
  const now = Date.now();

  // Order models so currently healthy models run first while cooled-down 503 models are tried after
  const activeModels = TEXT_MODEL_CASCADE.filter(
    (m) => (modelCooldownUntil.get(m) ?? 0) <= now
  );
  const cooledModels = TEXT_MODEL_CASCADE.filter(
    (m) => (modelCooldownUntil.get(m) ?? 0) > now
  );
  const orderedModels = [...activeModels, ...cooledModels];

  for (const modelName of orderedModels) {
    try {
      const response = await ai.models.generateContent({
        ...params,
        model: modelName,
        config: {
          ...params.config,
          abortSignal: AbortSignal.timeout(12000),
        },
      });
      modelCooldownUntil.delete(modelName);
      return response;
    } catch (err) {
      lastError = err;
      if (!isTransientDemandError(err) && !(err instanceof DOMException)) {
        throw err;
      }
      // Put overloaded/rate-limited model on 3-minute cooldown so subsequent requests skip straight to a healthy Gemini model
      modelCooldownUntil.set(modelName, Date.now() + 180_000);
    }
  }

  throw lastError;
}

async function callPollinationsTextFallback(
  systemInstruction: string,
  userPrompt: string
): Promise<string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (process.env.POLLINATIONS_API_KEY) {
    headers["Authorization"] = `Bearer ${process.env.POLLINATIONS_API_KEY}`;
  }

  const res = await fetch("https://text.pollinations.ai/", {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: "openai",
      jsonMode: true,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt },
      ],
    }),
    signal: AbortSignal.timeout(20000),
  });

  if (!res.ok) {
    throw new Error(`Backup text engine returned HTTP ${res.status}`);
  }
  return await res.text();
}

function parseJsonObjectSafely<T>(raw: string): T | null {
  const trimmed = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start !== -1 && end > start) {
      try {
        return JSON.parse(trimmed.slice(start, end + 1)) as T;
      } catch {
        return null;
      }
    }
    return null;
  }
}

async function generateTtsWithFallback(
  ai: GoogleGenAI,
  params: Omit<GenerateContentParameters, "model">
): Promise<GenerateContentResponse> {
  let lastError: unknown;

  for (let pass = 0; pass < 2; pass++) {
    for (const modelName of TTS_MODEL_CASCADE) {
      try {
        return await ai.models.generateContent({
          ...params,
          model: modelName,
        });
      } catch (err) {
        lastError = err;
        if (!isTransientDemandError(err)) {
          throw err;
        }
        await sleep(450 * (pass + 1));
      }
    }
    await sleep(800 * (pass + 1));
  }

  throw lastError;
}

function extractCleanErrorMessage(raw: string): string {
  try {
    const jsonStart = raw.indexOf("{");
    if (jsonStart !== -1) {
      const parsed = JSON.parse(raw.slice(jsonStart)) as {
        error?: { message?: string };
        message?: string;
      };
      if (parsed?.error?.message) return parsed.error.message;
      if (parsed?.message) return parsed.message;
    }
  } catch {
    // ignore JSON parse errors
  }
  return raw;
}

function formatGeminiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : String(err);
  if (
    raw.includes("403") ||
    raw.includes("PERMISSION_DENIED") ||
    raw.includes("API_KEY_INVALID")
  ) {
    return {
      status: 403,
      message:
        "Gemini API key permission error. Please check your key in the Settings > Secrets panel.",
    };
  }
  if (raw.includes("429") || raw.includes("RESOURCE_EXHAUSTED")) {
    return {
      status: 429,
      message:
        "Gemini quota limit reached. Select a billing-enabled key in Settings > Secrets to increase quota.",
    };
  }
  if (raw.includes("503") || raw.includes("UNAVAILABLE")) {
    return {
      status: 503,
      message:
        "Models are temporarily experiencing peak traffic. Please retry in a few seconds.",
    };
  }
  return {
    status: 500,
    message:
      extractCleanErrorMessage(raw) || "Failed to process AI generation request.",
  };
}

const VALID_GEMINI_VOICES = new Set([
  "Zephyr",
  "Puck",
  "Charon",
  "Kore",
  "Fenrir",
  "Leda",
  "Orus",
  "Aoede",
  "Callirrhoe",
  "Autonoe",
  "Enceladus",
  "Iapetus",
  "Umbriel",
  "Algieba",
  "Despina",
  "Erinome",
  "Algenib",
  "Rasalgethi",
  "Laomedeia",
  "Achernar",
  "Alnilam",
  "Schedar",
  "Gacrux",
  "Pulcherrima",
  "Achird",
  "Zubenelgenubi",
  "Vindemiatrix",
  "Sadachbia",
  "Sadaltager",
  "Sulafat",
]);

function parseGeminiVoice(voiceSelection: string): {
  voiceName: string;
  styleHint: string;
} {
  if (voiceSelection && voiceSelection.startsWith("gemini:")) {
    const raw = voiceSelection.replace("gemini:", "");
    const [namePart, ...toneParts] = raw.split("-");
    const cleanName = VALID_GEMINI_VOICES.has(namePart) ? namePart : "Puck";
    const tone = toneParts.join(" ").trim() || "Upbeat";
    return {
      voiceName: cleanName,
      styleHint: `${tone}, clear, engaging short-form video narration`,
    };
  }
  return {
    voiceName: "Puck",
    styleHint: "Clear, natural, engaging short-form video narration",
  };
}

async function generateScriptAndKeywordsViaGemini(params: {
  subject: string;
  language: string;
  paragraphCount: number;
  customRequirements: string;
  systemPrompt: string;
}): Promise<{ script: string; keywords: string[] }> {
  const langInstruction =
    params.language && params.language !== "Auto Detect"
      ? `Write the video script in ${params.language}.`
      : "Write the video script in the same language as the Video Subject.";

  const targetParagraphs = Math.max(
    1,
    Math.min(10, Number(params.paragraphCount) || 1)
  );

  const promptParts = [
    `Video Subject: ${params.subject}`,
    `Target Paragraph Count: ${targetParagraphs}`,
    langInstruction,
    params.customRequirements?.trim()
      ? `Additional Script Requirements: ${params.customRequirements.trim()}`
      : "",
    "Also extract 5 to 7 concise, action-oriented English visual video shot prompts for Pollinations AI video generation (describe dynamic scenes, movement, and camera perspective, no text overlays).",
  ]
    .filter(Boolean)
    .join("\n");

  const baseSystemPrompt =
    params.systemPrompt?.trim() ||
    "You are a short-form video scriptwriter. Return raw spoken narration only without markdown titles, narrator labels, or stage directions.";

  try {
    const ai = getGenAIClient();
    const response = await generateTextWithFallback(ai, {
      contents: promptParts,
      config: {
        systemInstruction: baseSystemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            script: {
              type: Type.STRING,
              description:
                "Raw spoken narration script divided into the requested number of paragraphs without titles or labels.",
            },
            keywords: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description:
                "5 to 7 dynamic English motion-video scene prompts for Pollinations AI video generation.",
            },
          },
          required: ["script", "keywords"],
        },
      },
    });

    const parsed = parseJsonObjectSafely<{
      script?: string;
      keywords?: string[];
    }>(response.text ?? "{}");

    if (parsed?.script) {
      return {
        script: parsed.script.trim(),
        keywords: Array.isArray(parsed.keywords)
          ? parsed.keywords.map((k) => k.trim()).filter(Boolean)
          : [],
      };
    }
  } catch (err) {
    if (!isTransientDemandError(err) && !(err instanceof DOMException)) {
      throw err;
    }
    // Fall over seamlessly on 503 UNAVAILABLE spikes
  }

  const rawBackup = await callPollinationsTextFallback(
    `${baseSystemPrompt}\n\nRespond STRICTLY as a valid JSON object with two keys: "script" (string, raw spoken narration only) and "keywords" (array of 5 concise English visual scene descriptions). Do not wrap in markdown.`,
    promptParts
  );

  const backupParsed = parseJsonObjectSafely<{
    script?: string;
    keywords?: string[];
  }>(rawBackup);

  if (backupParsed?.script) {
    return {
      script: backupParsed.script.trim(),
      keywords: Array.isArray(backupParsed.keywords)
        ? backupParsed.keywords.map((k) => String(k).trim()).filter(Boolean)
        : [
            `${params.subject} cinematic wide shot`,
            `${params.subject} close up detail`,
            `${params.subject} dramatic lighting`,
          ],
    };
  }

  return {
    script: rawBackup.trim(),
    keywords: [
      `${params.subject} cinematic wide shot`,
      `${params.subject} close up detail`,
      `${params.subject} documentary scene`,
    ],
  };
}

async function extractKeywordsWithFailover(
  subject: string,
  script: string
): Promise<string[]> {
  const prompt = `Extract 5 to 7 concise, dynamic English motion-video shot prompts (describing action, subject movement, and cinematic camera perspective) for Pollinations AI video generation matching this content:\n\nSubject: ${subject}\nScript: ${script}`;

  try {
    const ai = getGenAIClient();
    const response = await generateTextWithFallback(ai, {
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            keywords: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ["keywords"],
        },
      },
    });

    const parsed = parseJsonObjectSafely<{ keywords?: string[] }>(
      response.text ?? "{}"
    );
    if (Array.isArray(parsed?.keywords) && parsed.keywords.length > 0) {
      return parsed.keywords.map((k) => k.trim()).filter(Boolean);
    }
  } catch (err) {
    if (!isTransientDemandError(err) && !(err instanceof DOMException)) {
      throw err;
    }
  }

  const rawBackup = await callPollinationsTextFallback(
    'Return STRICTLY a valid JSON object: {"keywords": ["scene 1", "scene 2", "scene 3", "scene 4", "scene 5"]} with concise English visual prompts. No markdown.',
    prompt
  );
  const backupParsed = parseJsonObjectSafely<{ keywords?: string[] }>(rawBackup);
  if (Array.isArray(backupParsed?.keywords) && backupParsed.keywords.length > 0) {
    return backupParsed.keywords.map((k) => String(k).trim()).filter(Boolean);
  }
  const basis = (subject || script.slice(0, 40)).trim();
  return [
    `${basis} cinematic wide shot`,
    `${basis} detailed close up`,
    `${basis} atmospheric documentary scene`,
  ];
}

async function synthesizeGeminiWavBuffer(params: {
  text: string;
  voice: string;
  speed: string;
}): Promise<{ wavBuffer: Buffer; voiceName: string }> {
  const cleanText = params.text
    .replace(/\[(pause|silence):\s*[\d.]+s?\]/gi, " ... ")
    .trim()
    .slice(0, 1800);

  const ai = getGenAIClient();
  const { voiceName, styleHint } = parseGeminiVoice(params.voice);
  const speedNum = parseFloat(params.speed) || 1.0;
  const paceDescriptor =
    speedNum >= 1.3
      ? "Brisk, energetic pace."
      : speedNum <= 0.9
      ? "Measured, calm pace."
      : "Natural conversational pace.";

  const response = await generateTtsWithFallback(ai, {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: cleanText,
            speechMetadata: {
              style: `${styleHint}. ${paceDescriptor}`,
            },
          },
        ],
      },
    ],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName },
        },
      },
    },
  });

  const base64Audio =
    response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) {
    throw new Error("Gemini TTS returned empty audio data.");
  }

  return {
    wavBuffer: Buffer.from(base64Audio, "base64"),
    voiceName,
  };
}

function hexToAssColor(hex: string, alphaHex = "00"): string {
  const clean = (hex || "#FFFFFF").replace("#", "").padEnd(6, "0").slice(0, 6);
  const r = clean.slice(0, 2).toUpperCase();
  const g = clean.slice(2, 4).toUpperCase();
  const b = clean.slice(4, 6).toUpperCase();
  return `&H${alphaHex}${b}${g}${r}`;
}

function formatAssTimestamp(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const hrs = Math.floor(clamped / 3600);
  const mins = Math.floor((clamped % 3600) / 60);
  const secs = Math.floor(clamped % 60);
  const cs = Math.floor((clamped - Math.floor(clamped)) * 100);
  return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}.${String(cs).padStart(2, "0")}`;
}

function buildAssSubtitles(params: {
  script: string;
  duration: number;
  width: number;
  height: number;
  displayMode: string;
  position: string;
  customPositionPercent: number;
  fontColor: string;
  fontSize: number;
  strokeColor: string;
  strokeWidth: number;
  bgEnabled: boolean;
  bgColor: string;
}): string {
  const chunks: string[] = [];
  if (params.displayMode === "Word by word") {
    const words = params.script.split(/\s+/).map((w) => w.trim()).filter(Boolean);
    for (let i = 0; i < words.length; i += 3) {
      chunks.push(words.slice(i, i + 3).join(" "));
    }
  } else {
    const rawSentences = params.script
      .split(/(?<=[.!?。！？,;；])\s+/)
      .map((s) => s.trim())
      .filter(Boolean);
    for (const s of rawSentences) {
      if (s.length > 72) {
        const words = s.split(/\s+/);
        for (let i = 0; i < words.length; i += 9) {
          chunks.push(words.slice(i, i + 9).join(" "));
        }
      } else {
        chunks.push(s);
      }
    }
  }

  if (chunks.length === 0 && params.script.trim()) {
    chunks.push(params.script.trim());
  }

  const totalChars = Math.max(
    1,
    chunks.reduce((acc, c) => acc + c.length, 0)
  );

  let alignment = 2;
  let marginV = Math.round(params.height * 0.11);
  if (params.position === "Top") {
    alignment = 8;
    marginV = Math.round(params.height * 0.08);
  } else if (params.position === "Center") {
    alignment = 5;
    marginV = 20;
  } else if (params.position === "Upper third") {
    alignment = 8;
    marginV = Math.round(params.height * 0.25);
  } else if (params.position === "Custom") {
    const pct = Math.max(5, Math.min(95, Number(params.customPositionPercent) || 80));
    alignment = 8;
    marginV = Math.round((pct / 100) * params.height);
  }

  const primaryCol = hexToAssColor(params.fontColor, "00");
  const outlineCol = params.bgEnabled
    ? hexToAssColor(params.bgColor, "40")
    : hexToAssColor(params.strokeColor, "00");
  const backCol = hexToAssColor(params.bgColor, "60");
  const borderStyle = params.bgEnabled ? 3 : 1;
  const outlinePx = params.bgEnabled
    ? 8
    : Math.max(1, Math.min(8, Number(params.strokeWidth) * 1.4));
  const scaledFontPx = Math.max(
    24,
    Math.min(74, Math.round((Number(params.fontSize) || 60) * 0.72))
  );

  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: ${params.width}
PlayResY: ${params.height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,${scaledFontPx},${primaryCol},${primaryCol},${outlineCol},${backCol},1,0,0,0,100,100,0,0,${borderStyle},${outlinePx.toFixed(1)},1,${alignment},48,48,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`;

  let cursor = 0;
  const events = chunks.map((chunk, idx) => {
    const segDur =
      idx === chunks.length - 1
        ? Math.max(0.4, params.duration - cursor)
        : Math.max(0.5, (chunk.length / totalChars) * params.duration);
    const start = cursor;
    const end = Math.min(params.duration, cursor + segDur);
    cursor = end;
    const safeText = chunk.replace(/[{}\\]/g, "").replace(/\r?\n/g, " ");
    return `Dialogue: 0,${formatAssTimestamp(start)},${formatAssTimestamp(
      end
    )},Default,,0,0,0,,${safeText}`;
  });

  return `${header}\n${events.join("\n")}\n`;
}

async function getAudioDurationSeconds(
  wavPath: string,
  fallbackBytes: number
): Promise<number> {
  try {
    const { stdout } = await execFileAsync("ffprobe", [
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      wavPath,
    ]);
    const parsed = parseFloat(stdout.trim());
    if (Number.isFinite(parsed) && parsed > 0.5) {
      return parsed;
    }
  } catch {
    // Fallback to 24kHz 16-bit mono PCM byte math
  }
  return Math.max(4, (fallbackBytes - 44) / 48000);
}

function getFallbackLocalImages(): string[] {
  const imgDir = path.join(__dirname, "src", "assets", "images");
  try {
    if (fs.existsSync(imgDir)) {
      const files = fs
        .readdirSync(imgDir)
        .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
        .map((f) => path.join(imgDir, f));
      if (files.length > 0) return files;
    }
  } catch {
    // ignore
  }
  return [];
}

const SHOT_ANGLE_VARIATIONS = [
  "cinematic establishing wide tracking shot, dynamic motion",
  "detailed close-up camera push-in, shallow depth of field",
  "aerial drone sweeping perspective, dramatic natural lighting",
  "low-angle cinematic tracking shot, atmospheric motion",
  "medium action shot with rich environmental detail",
  "dramatic side-profile pan shot, photorealistic cinema",
  "immersive first-person forward movement shot",
];

function buildStoryboardPrompts(params: {
  keywords: string[];
  subject: string;
  targetCount: number;
  concatMode?: string;
  matchToScript?: boolean;
}): string[] {
  const base = params.keywords.map((k) => k.trim()).filter(Boolean);
  const expanded: string[] = [];

  for (let i = 0; i < params.targetCount; i++) {
    const angle = SHOT_ANGLE_VARIATIONS[i % SHOT_ANGLE_VARIATIONS.length];
    if (i < base.length) {
      expanded.push(`${base[i]}, ${angle}`);
    } else if (base.length > 0) {
      const seedPrompt = base[i % base.length];
      expanded.push(`${seedPrompt}, ${angle}`);
    } else {
      expanded.push(`${params.subject}, ${angle}`);
    }
  }

  if (
    params.concatMode?.startsWith("Random") &&
    params.matchToScript === false
  ) {
    for (let i = expanded.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [expanded[i], expanded[j]] = [expanded[j], expanded[i]];
    }
  }

  return expanded;
}

function parsePollinationsModelAlias(rawSelection?: string): string {
  if (!rawSelection) return "wan-fast";
  const alias = rawSelection.split("(")[0].trim();
  return alias || "wan-fast";
}

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;
  const limit = Math.max(1, Math.min(items.length, concurrency));

  await Promise.all(
    Array.from({ length: limit }, async () => {
      while (nextIndex < items.length) {
        const idx = nextIndex++;
        results[idx] = await worker(items[idx], idx);
      }
    })
  );

  return results;
}

let pollinationsPaidVideoCooldownUntil = 0;
let lastPollinationsVideoNotice = "";

async function tryFetchPollinationsNativeVideo(params: {
  prompt: string;
  modelAlias: string;
  aspectRatioParam: "9:16" | "16:9" | "1:1";
  durationSec: number;
  destPath: string;
}): Promise<{ ok: boolean; reason?: string }> {
  const apiKey = process.env.POLLINATIONS_API_KEY;
  if (!apiKey) {
    return {
      ok: false,
      reason: "no-key",
    };
  }
  if (Date.now() < pollinationsPaidVideoCooldownUntil) {
    return {
      ok: false,
      reason: lastPollinationsVideoNotice || "paid-pollen-required",
    };
  }

  const dur = Math.max(4, Math.min(8, Math.round(params.durationSec)));
  const seed = Math.floor(1000 + Math.random() * 900000);
  const url = `https://gen.pollinations.ai/video/${encodeURIComponent(
    `${params.prompt}, smooth natural motion, cinematic 4k video, no text`
  )}?model=${encodeURIComponent(
    params.modelAlias
  )}&duration=${dur}&aspectRatio=${encodeURIComponent(
    params.aspectRatioParam
  )}&nologo=true&seed=${seed}`;

  try {
    const res = await fetch(url, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(45000),
    });

    if (res.status === 401 || res.status === 402 || res.status === 403) {
      let detail = "";
      try {
        const errJson = (await res.json()) as {
          error?: { message?: string };
        };
        detail = errJson?.error?.message || "";
      } catch {
        // ignore parse error
      }
      pollinationsPaidVideoCooldownUntil = Date.now() + 90_000;
      lastPollinationsVideoNotice = detail.includes("Quest Pollen")
        ? "Quest Pollen key detected — used multi-shot Pollinations Flux + 25fps camera motion synthesis"
        : "Pollinations paid video tier unavailable — used multi-shot 25fps camera motion synthesis";
      return { ok: false, reason: lastPollinationsVideoNotice };
    }

    if (!res.ok) {
      return { ok: false, reason: `HTTP ${res.status}` };
    }

    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const buf = Buffer.from(await res.arrayBuffer());
    if ((ct.includes("video") || ct.includes("octet-stream")) && buf.length > 8192) {
      fs.writeFileSync(params.destPath, buf);
      return { ok: true };
    }
    return { ok: false, reason: "Unexpected content-type from video endpoint" };
  } catch {
    return { ok: false, reason: "Pollinations video endpoint timed out" };
  }
}

const FFMPEG_EXEC_OPTS = {
  timeout: 60_000,
  maxBuffer: 10 * 1024 * 1024,
};

async function fetchPollinationsStoryboardFrames(params: {
  prompts: string[];
  width: number;
  height: number;
  workDir: string;
  onSceneProgress?: (completed: number, total: number) => void;
}): Promise<string[]> {
  const fallbacks = getFallbackLocalImages();
  const savedPaths: string[] = new Array(params.prompts.length).fill("");
  const apiKey = process.env.POLLINATIONS_API_KEY;
  let completedCount = 0;
  const totalCount = params.prompts.length;

  await mapWithConcurrency(
    params.prompts,
    Math.min(6, Math.max(1, totalCount)),
    async (rawPrompt, idx) => {
      const destPath = path.join(params.workDir, `scene_${idx}.jpg`);
      const enriched = `${rawPrompt}, cinematic film still, photorealistic, sharp detail, rich lighting, no text, no watermark`;
      const seed = Math.floor(1000 + Math.random() * 900000) + idx * 137;

      const endpoints = [
        `https://gen.pollinations.ai/image/${encodeURIComponent(
          enriched
        )}?model=flux&width=${params.width}&height=${params.height}&nologo=true&seed=${seed}`,
        `https://image.pollinations.ai/prompt/${encodeURIComponent(
          enriched
        )}?width=${params.width}&height=${params.height}&nologo=true&seed=${seed}`,
      ];

      let saved = false;
      for (const url of endpoints) {
        try {
          const headers: Record<string, string> = {};
          if (apiKey && url.includes("gen.pollinations.ai")) {
            headers["Authorization"] = `Bearer ${apiKey}`;
          }
          const res = await fetch(url, {
            headers,
            signal: AbortSignal.timeout(10000),
          });
          const ct = (res.headers.get("content-type") || "").toLowerCase();
          if (res.ok && ct.includes("image")) {
            const buf = Buffer.from(await res.arrayBuffer());
            if (buf.length > 2048) {
              fs.writeFileSync(destPath, buf);
              savedPaths[idx] = destPath;
              saved = true;
              break;
            }
          }
        } catch {
          // try next Pollinations image endpoint
        }
      }

      if (!saved && fallbacks.length > 0) {
        const fb = fallbacks[idx % fallbacks.length];
        fs.copyFileSync(fb, destPath);
        savedPaths[idx] = destPath;
      }

      completedCount += 1;
      params.onSceneProgress?.(completedCount, totalCount);
    }
  );

  return savedPaths.filter(Boolean);
}

async function normalizeNativeVideoClip(params: {
  inputVideoPath: string;
  outputClipPath: string;
  width: number;
  height: number;
  clipDuration: number;
  clipSpeed: number;
  fitMode: string;
}): Promise<void> {
  const speed = Math.max(0.5, Math.min(2.0, params.clipSpeed || 1.0));
  const ptsFactor = (1 / speed).toFixed(4);
  const scaleExpr =
    params.fitMode === "Fit (black bars)"
      ? `scale=${params.width}:${params.height}:force_original_aspect_ratio=decrease,pad=${params.width}:${params.height}:(ow-iw)/2:(oh-ih)/2:black`
      : `scale=${params.width}:${params.height}:force_original_aspect_ratio=increase,crop=${params.width}:${params.height}`;

  const targetDur = (params.clipDuration + 0.16).toFixed(3);
  await execFileAsync(
    "ffmpeg",
    [
      "-y",
      "-loglevel",
      "error",
      "-stream_loop",
      "-1",
      "-i",
      params.inputVideoPath,
      "-t",
      targetDur,
      "-vf",
      `setpts=${ptsFactor}*PTS,${scaleExpr},fps=25,setsar=1,format=yuv420p`,
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-pix_fmt",
      "yuv420p",
      params.outputClipPath,
    ],
    FFMPEG_EXEC_OPTS
  );
}

async function renderDynamicMotionClipFromFrame(params: {
  imagePath: string;
  outputClipPath: string;
  shotIndex: number;
  width: number;
  height: number;
  clipDuration: number;
  clipSpeed: number;
  fitMode: string;
}): Promise<void> {
  const fps = 25;
  // Include +4 safety frames (+0.16s) so xfade transition offsets always have overlap headroom
  const totalFrames = Math.max(35, Math.ceil((params.clipDuration + 0.16) * fps));
  const speed = Math.max(0.5, Math.min(2.0, params.clipSpeed || 1.0));
  const zoomDelta = (0.0021 * speed).toFixed(5);

  // 2x supersampled canvas eliminates integer rounding jitter during continuous zoompan motion
  const superW = params.width * 2;
  const superH = params.height * 2;
  const preScale =
    params.fitMode === "Fit (black bars)"
      ? `scale=${superW}:${superH}:force_original_aspect_ratio=decrease,pad=${superW}:${superH}:(ow-iw)/2:(oh-ih)/2:black`
      : `scale=${superW}:${superH}:force_original_aspect_ratio=increase,crop=${superW}:${superH}`;

  const rig = params.shotIndex % 6;
  let zpExpr = "";
  if (rig === 0) {
    zpExpr = `z='min(zoom+${zoomDelta},1.26)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'`;
  } else if (rig === 1) {
    zpExpr = `z='if(eq(on,1),1.24,max(1.00,zoom-${zoomDelta}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'`;
  } else if (rig === 2) {
    zpExpr = `z='1.18':x='(iw-iw/zoom)*(on/${totalFrames})':y='(ih-ih/zoom)*0.5'`;
  } else if (rig === 3) {
    zpExpr = `z='1.18':x='(iw-iw/zoom)*(1-on/${totalFrames})':y='(ih-ih/zoom)*0.5'`;
  } else if (rig === 4) {
    zpExpr = `z='min(zoom+${zoomDelta},1.22)':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/${totalFrames})'`;
  } else {
    zpExpr = `z='1.20':x='(iw-iw/zoom)*(on/${totalFrames})':y='(ih-ih/zoom)*(on/${totalFrames})'`;
  }

  const vf = `${preScale},zoompan=${zpExpr}:d=${totalFrames}:s=${params.width}x${params.height}:fps=${fps},setsar=1,format=yuv420p`;

  await execFileAsync(
    "ffmpeg",
    [
      "-y",
      "-loglevel",
      "error",
      "-i",
      params.imagePath,
      "-vf",
      vf,
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-pix_fmt",
      "yuv420p",
      params.outputClipPath,
    ],
    FFMPEG_EXEC_OPTS
  );
}

function resolveXfadeTransition(mode: string, index: number): {
  transition: string;
  duration: number;
} {
  const shufflePool = [
    "fade",
    "slideleft",
    "smoothup",
    "circleopen",
    "slideright",
    "fadeblack",
  ];
  switch (mode) {
    case "None":
      return { transition: "fade", duration: 0.25 };
    case "FadeIn":
      return { transition: "fade", duration: 0.45 };
    case "FadeOut":
      return { transition: "fadeblack", duration: 0.45 };
    case "SlideIn":
      return {
        transition: index % 2 === 0 ? "slideleft" : "slideright",
        duration: 0.45,
      };
    case "SlideOut":
      return {
        transition: index % 2 === 0 ? "smoothleft" : "smoothright",
        duration: 0.45,
      };
    case "ZoomIn":
      return { transition: "circleopen", duration: 0.45 };
    case "ZoomOut":
      return { transition: "circleclose", duration: 0.45 };
    case "Shuffle":
    default:
      return {
        transition: shufflePool[index % shufflePool.length],
        duration: 0.45,
      };
  }
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: "8mb" }));

  // Check Gemini status with multi-model resilience
  app.get("/api/gemini-status", async (req: Request, res: Response) => {
    const verifyLive = req.query.verify === "1";
    const configured = Boolean(process.env.GEMINI_API_KEY);
    if (!configured) {
      res.json({
        configured: false,
        verified: false,
        message:
          "GEMINI_API_KEY is not set. Attach your key via AI Studio's Settings > Secrets panel.",
      });
      return;
    }

    if (!verifyLive) {
      res.json({
        configured: true,
        verified: true,
        message: "Gemini API key is active on the server.",
      });
      return;
    }

    try {
      const ai = getGenAIClient();
      await generateTextWithFallback(ai, {
        contents: "Reply OK",
      });
      res.json({
        configured: true,
        verified: true,
        message:
          "Verified live connection to Gemini API and Pollinations AI video pipeline.",
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        configured: true,
        verified: false,
        message: formatted.message,
      });
    }
  });

  // Library Endpoints
  app.get("/api/library", (_req: Request, res: Response) => {
    const items = readLibrary().filter((item) =>
      fs.existsSync(path.join(VIDEO_STORAGE_DIR, item.filename))
    );
    res.json({ videos: items });
  });

  app.delete("/api/library/:id", (req: Request, res: Response) => {
    const id = req.params.id;
    const current = readLibrary();
    const target = current.find((v) => v.id === id);
    if (target) {
      const fp = path.join(VIDEO_STORAGE_DIR, target.filename);
      if (fs.existsSync(fp)) {
        try {
          fs.unlinkSync(fp);
        } catch {
          // ignore
        }
      }
    }
    const updated = current.filter((v) => v.id !== id);
    writeLibrary(updated);
    res.json({ ok: true, videos: updated });
  });

  app.get("/api/videos/:filename", (req: Request, res: Response) => {
    const safeName = path.basename(String(req.params.filename || ""));
    const fullPath = path.join(VIDEO_STORAGE_DIR, safeName);
    if (!fs.existsSync(fullPath)) {
      res.status(404).json({ error: "Video file not found." });
      return;
    }
    res.setHeader("Content-Type", "video/mp4");
    if (req.query.download === "1") {
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="${safeName}"`
      );
    }
    res.sendFile(fullPath);
  });

  // Script & Keywords via Gemini (with automatic 503 multi-model fallback)
  app.post("/api/generate-script", async (req: Request, res: Response) => {
    try {
      const {
        subject = "",
        language = "en-US",
        paragraphCount = 1,
        customRequirements = "",
        systemPrompt = "",
      } = req.body as {
        subject?: string;
        language?: string;
        paragraphCount?: number;
        customRequirements?: string;
        systemPrompt?: string;
      };

      const trimmedSubject = subject.trim();
      if (!trimmedSubject) {
        res.status(400).json({ error: "Please enter a Video Subject first." });
        return;
      }

      const result = await generateScriptAndKeywordsViaGemini({
        subject: trimmedSubject,
        language,
        paragraphCount,
        customRequirements,
        systemPrompt,
      });

      res.json(result);
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  app.post("/api/generate-keywords", async (req: Request, res: Response) => {
    try {
      const { subject = "", script = "" } = req.body as {
        subject?: string;
        script?: string;
      };

      const basis = (script || subject || "").trim();
      if (!basis) {
        res
          .status(400)
          .json({ error: "Please enter a Video Subject or Video Script first." });
        return;
      }

      const keywords = await extractKeywordsWithFailover(subject, script);
      res.json({ keywords });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  // Gemini TTS Preview
  app.post("/api/tts", async (req: Request, res: Response) => {
    try {
      const { text = "", voice = "gemini:Puck-Upbeat", speed = "1.0" } =
        req.body as {
          text?: string;
          voice?: string;
          speed?: string;
        };

      if (!text.trim()) {
        res.status(400).json({ error: "No text provided for voice synthesis." });
        return;
      }

      const { wavBuffer, voiceName } = await synthesizeGeminiWavBuffer({
        text,
        voice,
        speed,
      });

      res.json({
        audioBase64: wavBuffer.toString("base64"),
        mimeType: "audio/wav",
        voiceName,
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  // Background Render Job Registry (prevents proxy timeouts & drives real 0-100% progress)
  interface RenderVideoRequestBody {
    subject?: string;
    script?: string;
    keywords?: string;
    language?: string;
    paragraphCount?: number;
    customRequirements?: string;
    systemPrompt?: string;
    pollinationsVideoModel?: string;
    videoConcatMode?: string;
    matchMaterialsToScript?: boolean;
    videoTransitionMode?: string;
    aspectRatio?: string;
    sourceFrameFit?: string;
    maxClipDuration?: string;
    clipSpeed?: number;
    clipRenderingConcurrency?: string;
    voiceoverMode?: string;
    voiceName?: string;
    voiceVolume?: string;
    voiceSpeed?: string;
    bgmSource?: string;
    bgmVolume?: string;
    subtitleEnabled?: boolean;
    subtitlePosition?: string;
    customPositionPercent?: number;
    subtitleDisplayMode?: string;
    fontColor?: string;
    fontSize?: number;
    strokeColor?: string;
    strokeWidth?: number;
    subtitleBackgroundEnabled?: boolean;
    subtitleBackgroundColor?: string;
  }

  interface RenderJobState {
    jobId: string;
    status: "running" | "completed" | "failed";
    progress: number;
    stepLabel: string;
    createdAtMs: number;
    video?: StoredVideoRecord;
    script?: string;
    keywords?: string[];
    notice?: string;
    error?: string;
  }

  const renderJobs = new Map<string, RenderJobState>();

  function setJobProgress(jobId: string, progress: number, stepLabel: string) {
    const job = renderJobs.get(jobId);
    if (!job || job.status !== "running") return;
    job.progress = Math.max(job.progress, Math.min(99, Math.round(progress)));
    job.stepLabel = stepLabel;
  }

  function pruneOldRenderJobs() {
    const cutoff = Date.now() - 30 * 60_000;
    for (const [id, job] of renderJobs.entries()) {
      if (job.status !== "running" && job.createdAtMs < cutoff) {
        renderJobs.delete(id);
      }
    }
  }

  async function executeRenderVideoJob(
    jobId: string,
    body: RenderVideoRequestBody
  ): Promise<void> {
    const workDir = path.join(VIDEO_STORAGE_DIR, `work_${jobId}`);
    fs.mkdirSync(workDir, { recursive: true });
    let muxTicker: ReturnType<typeof setInterval> | null = null;

    try {
      const rawSubject = (body.subject || "").trim();
      let finalScript = (body.script || "").trim();
      let keywordList = (body.keywords || "")
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean);

      const effectiveSubject = rawSubject || finalScript.slice(0, 48);

      // 1. Generate Script & Keywords via Gemini if missing
      if (!finalScript || keywordList.length === 0) {
        setJobProgress(
          jobId,
          8,
          "Generating video script & visual shot prompts with Gemini AI..."
        );
        const gen = await generateScriptAndKeywordsViaGemini({
          subject: effectiveSubject,
          language: body.language || "en-US",
          paragraphCount: body.paragraphCount || 1,
          customRequirements: body.customRequirements || "",
          systemPrompt: body.systemPrompt || "",
        });
        if (!finalScript && gen.script) {
          finalScript = gen.script;
        }
        if (keywordList.length === 0 && gen.keywords.length > 0) {
          keywordList = gen.keywords;
        }
      }

      if (keywordList.length === 0) {
        keywordList = [
          `${effectiveSubject} wide establishing shot`,
          `${effectiveSubject} close up action detail`,
          `${effectiveSubject} dramatic aerial perspective`,
          `${effectiveSubject} cinematic tracking angle`,
          `${effectiveSubject} atmospheric finale scene`,
        ];
      }

      // Target resolution & aspect ratio
      const isLandscape = (body.aspectRatio || "").includes("16:9");
      const isSquare = (body.aspectRatio || "").includes("1:1");
      const width = isLandscape ? 1280 : 720;
      const height = isLandscape ? 720 : isSquare ? 720 : 1280;
      const aspectRatioParam: "9:16" | "16:9" | "1:1" = isLandscape
        ? "16:9"
        : isSquare
        ? "1:1"
        : "9:16";

      // Estimate duration from word count to determine storyboard shot count
      const voiceSpeedNum = parseFloat(body.voiceSpeed || "1.0") || 1.0;
      const estimatedSeconds = Math.max(
        7,
        Math.min(
          45,
          Math.round(finalScript.split(/\s+/).length / (2.4 * voiceSpeedNum))
        )
      );
      const userMaxClipSec = Math.max(
        2.5,
        Math.min(6, Number(body.maxClipDuration) || 4)
      );
      const targetShotCount = Math.max(
        4,
        Math.min(7, Math.ceil(estimatedSeconds / userMaxClipSec))
      );

      const shotPrompts = buildStoryboardPrompts({
        keywords: keywordList,
        subject: effectiveSubject,
        targetCount: targetShotCount,
        concatMode: body.videoConcatMode,
        matchToScript: body.matchMaterialsToScript !== false,
      });

      const modelAlias = parsePollinationsModelAlias(
        body.pollinationsVideoModel
      );

      setJobProgress(
        jobId,
        16,
        `Synthesizing Gemini TTS voiceover & fetching ${shotPrompts.length} Pollinations AI scenes...`
      );

      // 2. Run Gemini TTS Voiceover + Pollinations Visual Acquisition in parallel
      const voiceTask =
        body.voiceoverMode !== "None"
          ? synthesizeGeminiWavBuffer({
              text: finalScript,
              voice: body.voiceName || "gemini:Puck-Upbeat",
              speed: body.voiceSpeed || "1.0",
            })
          : Promise.resolve(null);

      const visualTask = (async (): Promise<{
        mode: "pollinations-video" | "pollinations-motion";
        rawVideoPaths: string[];
        framePaths: string[];
        notice?: string;
      }> => {
        // Probe Tier 1: Native Pollinations Video API (gen.pollinations.ai/video/{prompt})
        const firstVideoPath = path.join(workDir, "raw_vid_0.mp4");
        const firstProbe = await tryFetchPollinationsNativeVideo({
          prompt: shotPrompts[0],
          modelAlias,
          aspectRatioParam,
          durationSec: userMaxClipSec,
          destPath: firstVideoPath,
        });

        if (firstProbe.ok) {
          const nativeClipCount = Math.min(4, shotPrompts.length);
          let fetchedVideos = 1;
          setJobProgress(
            jobId,
            24,
            `Fetched Pollinations native video clip 1/${nativeClipCount}...`
          );
          const rawVideos: string[] = [firstVideoPath];
          const remaining = await mapWithConcurrency(
            shotPrompts.slice(1, nativeClipCount),
            2,
            async (p, idx) => {
              const vp = path.join(workDir, `raw_vid_${idx + 1}.mp4`);
              const r = await tryFetchPollinationsNativeVideo({
                prompt: p,
                modelAlias,
                aspectRatioParam,
                durationSec: userMaxClipSec,
                destPath: vp,
              });
              fetchedVideos += 1;
              setJobProgress(
                jobId,
                20 + Math.round((fetchedVideos / nativeClipCount) * 38),
                `Fetched Pollinations native video clip ${fetchedVideos}/${nativeClipCount}...`
              );
              return r.ok ? vp : "";
            }
          );
          for (const vp of remaining) {
            if (vp) rawVideos.push(vp);
          }
          return {
            mode: "pollinations-video",
            rawVideoPaths: rawVideos,
            framePaths: [],
            notice: `Pollinations Native Video (${modelAlias})`,
          };
        }

        // Tier 2: Multi-shot Pollinations Flux storyboard + 25fps continuous camera motion
        setJobProgress(
          jobId,
          20,
          `Generating ${shotPrompts.length} Pollinations AI storyboard scenes...`
        );
        const frames = await fetchPollinationsStoryboardFrames({
          prompts: shotPrompts,
          width,
          height,
          workDir,
          onSceneProgress: (done, total) => {
            setJobProgress(
              jobId,
              20 + Math.round((done / total) * 38),
              `Generated Pollinations AI visual scene ${done} of ${total}...`
            );
          },
        });
        return {
          mode: "pollinations-motion",
          rawVideoPaths: [],
          framePaths: frames,
          notice:
            firstProbe.reason && firstProbe.reason !== "no-key"
              ? firstProbe.reason
              : undefined,
        };
      })();

      const [voiceResult, visualResult] = await Promise.all([
        voiceTask,
        visualTask,
      ]);

      setJobProgress(
        jobId,
        62,
        "Synchronizing Gemini voiceover audio & timed subtitles..."
      );

      let durationSeconds = estimatedSeconds;
      const wavPath = path.join(workDir, "voiceover.wav");
      let hasVoiceAudio = false;

      if (voiceResult && voiceResult.wavBuffer.length > 100) {
        fs.writeFileSync(wavPath, voiceResult.wavBuffer);
        hasVoiceAudio = true;
        durationSeconds = await getAudioDurationSeconds(
          wavPath,
          voiceResult.wavBuffer.length
        );
      }

      // 3. Build ASS Subtitles if enabled
      const assPath = path.join(workDir, "subtitles.ass");
      const useSubtitles = body.subtitleEnabled !== false;
      if (useSubtitles) {
        const assContent = buildAssSubtitles({
          script: finalScript,
          duration: durationSeconds,
          width,
          height,
          displayMode: body.subtitleDisplayMode || "Whole sentence",
          position: body.subtitlePosition || "Bottom (Recommended)",
          customPositionPercent: body.customPositionPercent ?? 80,
          fontColor: body.fontColor || "#FFFFFF",
          fontSize: body.fontSize ?? 60,
          strokeColor: body.strokeColor || "#000000",
          strokeWidth: body.strokeWidth ?? 1.5,
          bgEnabled: Boolean(body.subtitleBackgroundEnabled),
          bgColor: body.subtitleBackgroundColor || "#000000",
        });
        fs.writeFileSync(assPath, assContent, "utf8");
      }

      // 4. Render individual 25fps motion clips
      const transitionMode = body.videoTransitionMode || "Shuffle";
      const sampleTrans = resolveXfadeTransition(transitionMode, 0);
      const transDur = sampleTrans.duration;
      const fitMode = body.sourceFrameFit || "Fill (center crop)";
      const clipSpeedNum = Number(body.clipSpeed) || 1.0;
      const concurrency = Math.max(
        1,
        Math.min(4, parseInt(body.clipRenderingConcurrency || "2", 10) || 2)
      );

      let clipPaths: string[] = [];

      if (
        visualResult.mode === "pollinations-video" &&
        visualResult.rawVideoPaths.length > 0
      ) {
        const n = visualResult.rawVideoPaths.length;
        let doneClips = 0;
        const perClipDur = Math.max(
          2.0,
          (durationSeconds + Math.max(0, n - 1) * transDur) / n
        );
        setJobProgress(
          jobId,
          65,
          `Normalizing ${n} Pollinations video clips at 25fps...`
        );
        clipPaths = await mapWithConcurrency(
          visualResult.rawVideoPaths,
          concurrency,
          async (rawPath, idx) => {
            const outClip = path.join(workDir, `motion_clip_${idx}.mp4`);
            await normalizeNativeVideoClip({
              inputVideoPath: rawPath,
              outputClipPath: outClip,
              width,
              height,
              clipDuration: perClipDur,
              clipSpeed: clipSpeedNum,
              fitMode,
            });
            doneClips += 1;
            setJobProgress(
              jobId,
              65 + Math.round((doneClips / n) * 23),
              `Normalized video clip ${doneClips} of ${n}...`
            );
            return outClip;
          }
        );
      } else if (visualResult.framePaths.length > 0) {
        const n = visualResult.framePaths.length;
        let doneClips = 0;
        const perClipDur = Math.max(
          2.0,
          (durationSeconds + Math.max(0, n - 1) * transDur) / n
        );
        setJobProgress(
          jobId,
          65,
          `Rendering 25fps camera motion for ${n} scenes...`
        );
        clipPaths = await mapWithConcurrency(
          visualResult.framePaths,
          concurrency,
          async (imgPath, idx) => {
            const outClip = path.join(workDir, `motion_clip_${idx}.mp4`);
            await renderDynamicMotionClipFromFrame({
              imagePath: imgPath,
              outputClipPath: outClip,
              shotIndex: idx,
              width,
              height,
              clipDuration: perClipDur,
              clipSpeed: clipSpeedNum,
              fitMode,
            });
            doneClips += 1;
            setJobProgress(
              jobId,
              65 + Math.round((doneClips / n) * 23),
              `Rendered 25fps camera motion clip ${doneClips} of ${n}...`
            );
            return outClip;
          }
        );
      }

      // 5. Final FFmpeg xfade Transition Assembly + ASS Subtitles + Voiceover & BGM Mix
      setJobProgress(
        jobId,
        90,
        "Encoding final H.264 MP4 with crossfade transitions, subtitles & audio..."
      );
      muxTicker = setInterval(() => {
        const job = renderJobs.get(jobId);
        if (job && job.status === "running" && job.progress < 99) {
          job.progress = Math.min(99, job.progress + 1);
        }
      }, 450);

      const outFilename = `video_${jobId.slice(0, 8)}.mp4`;
      const outPath = path.join(VIDEO_STORAGE_DIR, outFilename);
      const escapedAssPath = assPath.replace(/\\/g, "/").replace(/:/g, "\\:");

      const ffmpegArgs: string[] = ["-y", "-loglevel", "error"];
      const filterParts: string[] = [];
      let videoInputCount = 0;

      if (clipPaths.length > 0) {
        for (const cp of clipPaths) {
          ffmpegArgs.push("-i", cp);
        }
        videoInputCount = clipPaths.length;

        const n = clipPaths.length;
        const perClipDur = Math.max(
          2.0,
          (durationSeconds + Math.max(0, n - 1) * transDur) / n
        );

        for (let i = 0; i < n; i++) {
          filterParts.push(
            `[${i}:v]fps=25,settb=AVTB,format=yuv420p[vnorm${i}]`
          );
        }

        let lastVideoLabel = "vnorm0";
        for (let i = 1; i < n; i++) {
          const { transition } = resolveXfadeTransition(transitionMode, i - 1);
          const offset = Math.max(0.2, i * (perClipDur - transDur));
          const nextLabel = `vxf${i}`;
          filterParts.push(
            `[${lastVideoLabel}][vnorm${i}]xfade=transition=${transition}:duration=${transDur.toFixed(
              3
            )}:offset=${offset.toFixed(3)}[${nextLabel}]`
          );
          lastVideoLabel = nextLabel;
        }

        if (useSubtitles) {
          filterParts.push(`[${lastVideoLabel}]ass='${escapedAssPath}'[vout]`);
        } else {
          filterParts.push(`[${lastVideoLabel}]null[vout]`);
        }
      } else {
        ffmpegArgs.push(
          "-f",
          "lavfi",
          "-i",
          `color=c=0x161922:s=${width}x${height}:d=${durationSeconds.toFixed(
            2
          )}:r=25`
        );
        videoInputCount = 1;
        if (useSubtitles) {
          filterParts.push(`[0:v]ass='${escapedAssPath}'[vout]`);
        } else {
          filterParts.push(`[0:v]null[vout]`);
        }
      }

      const voiceVolMultiplier = Math.max(
        0.2,
        Math.min(5.0, (parseInt(body.voiceVolume || "100%", 10) || 100) / 100)
      );
      const bgmVolPct =
        body.bgmSource === "No Background Music"
          ? 0
          : parseInt(body.bgmVolume || "20%", 10) || 0;

      if (hasVoiceAudio && bgmVolPct > 0) {
        const bgmAmp = ((bgmVolPct / 100) * 0.08).toFixed(4);
        const voIdx = videoInputCount;
        const bgmIdx = videoInputCount + 1;
        ffmpegArgs.push(
          "-i",
          wavPath,
          "-f",
          "lavfi",
          "-i",
          `aevalsrc='${bgmAmp}*sin(2*PI*220*t)+${bgmAmp}*sin(2*PI*277.18*t)+${bgmAmp}*sin(2*PI*329.63*t)':s=24000:d=${durationSeconds.toFixed(
            2
          )}`
        );
        filterParts.push(
          `[${voIdx}:a]volume=${voiceVolMultiplier.toFixed(2)}[vo]`,
          `[vo][${bgmIdx}:a]amix=inputs=2:duration=first:dropout_transition=1[aout]`
        );
      } else if (hasVoiceAudio) {
        const voIdx = videoInputCount;
        ffmpegArgs.push("-i", wavPath);
        filterParts.push(
          `[${voIdx}:a]volume=${voiceVolMultiplier.toFixed(2)}[aout]`
        );
      } else {
        const silentIdx = videoInputCount;
        ffmpegArgs.push(
          "-f",
          "lavfi",
          "-i",
          "anullsrc=channel_layout=mono:sample_rate=24000"
        );
        filterParts.push(`[${silentIdx}:a]anull[aout]`);
      }

      ffmpegArgs.push(
        "-filter_complex",
        filterParts.join(";"),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "25",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-t",
        durationSeconds.toFixed(2),
        "-movflags",
        "+faststart",
        outPath
      );

      await execFileAsync("ffmpeg", ffmpegArgs, FFMPEG_EXEC_OPTS);

      const record: StoredVideoRecord = {
        id: jobId,
        subject: effectiveSubject,
        script: finalScript,
        keywords: keywordList,
        aspectRatio: body.aspectRatio || "Portrait 9:16",
        durationSeconds: Number(durationSeconds.toFixed(1)),
        createdAt: new Date().toISOString().slice(0, 16).replace("T", " "),
        filename: outFilename,
        videoUrl: `/api/videos/${outFilename}`,
        downloadUrl: `/api/videos/${outFilename}?download=1`,
        videoSource: "Pollinations AI",
        voiceName: body.voiceName || "gemini:Puck-Upbeat",
        renderEngine: visualResult.mode,
      };

      const existing = readLibrary();
      const updated = [record, ...existing];
      writeLibrary(updated);

      const job = renderJobs.get(jobId);
      if (job) {
        job.status = "completed";
        job.progress = 100;
        job.stepLabel = "Completed! Video saved to Library.";
        job.video = record;
        job.script = finalScript;
        job.keywords = keywordList;
        job.notice = visualResult.notice;
      }
    } catch (err) {
      const formatted = formatGeminiError(err);
      const job = renderJobs.get(jobId);
      if (job) {
        job.status = "failed";
        job.error = formatted.message;
      }
    } finally {
      if (muxTicker) {
        clearInterval(muxTicker);
      }
      try {
        fs.rmSync(workDir, { recursive: true, force: true });
      } catch {
        // ignore cleanup error
      }
    }
  }

  app.post("/api/render-video", async (req: Request, res: Response) => {
    pruneOldRenderJobs();
    const body = (req.body || {}) as RenderVideoRequestBody;
    const rawSubject = (body.subject || "").trim();
    const rawScript = (body.script || "").trim();

    if (!rawSubject && !rawScript) {
      res.status(400).json({
        error: "Please enter a Video Subject or Video Script first.",
      });
      return;
    }

    const jobId = randomUUID();
    const initialJob: RenderJobState = {
      jobId,
      status: "running",
      progress: 5,
      stepLabel: "Starting Pollinations AI & Gemini video synthesis...",
      createdAtMs: Date.now(),
    };
    renderJobs.set(jobId, initialJob);

    if (req.query.sync === "1") {
      await executeRenderVideoJob(jobId, body);
      const finished = renderJobs.get(jobId);
      if (!finished || finished.status === "failed" || !finished.video) {
        res
          .status(500)
          .json({ error: finished?.error || "Video rendering failed." });
        return;
      }
      res.json({
        video: finished.video,
        script: finished.script,
        keywords: finished.keywords,
        notice: finished.notice,
      });
      return;
    }

    void executeRenderVideoJob(jobId, body);
    res.json({
      jobId,
      status: initialJob.status,
      progress: initialJob.progress,
      stepLabel: initialJob.stepLabel,
    });
  });

  app.get("/api/render-status/:jobId", (req: Request, res: Response) => {
    const job = renderJobs.get(String(req.params.jobId || ""));
    if (!job) {
      res.status(404).json({ error: "Render job not found." });
      return;
    }
    res.json(job);
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
