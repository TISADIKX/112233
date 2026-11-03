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

const VIDEO_RENDER_FPS = 30;

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

function isExplicitNonPhotoStyleRequested(
  subject: string,
  customRequirements = ""
): boolean {
  const combined = `${subject} ${customRequirements}`.toLowerCase();
  return /\b(cartoon|anime|manga|comic|pixel\s*art|vector\s*art|flat\s*illustration|2d\s*animation|crayon|watercolor|cel[- ]shaded)\b/i.test(
    combined
  );
}

function buildPhotorealisticStyleDescriptor(
  subject = "",
  customRequirements = ""
): {
  enforcePhoto: boolean;
  cameraModifier: string;
  negativePrompt: string;
} {
  if (isExplicitNonPhotoStyleRequested(subject, customRequirements)) {
    return {
      enforcePhoto: false,
      cameraModifier:
        "high detail, sharp focus, natural composition, no text, no watermark",
      negativePrompt:
        "text, watermark, logo, blurry, deformed, distorted, low quality",
    };
  }

  return {
    enforcePhoto: true,
    cameraModifier:
      "RAW live-action cinema photograph, shot on ARRI Alexa 35mm prime lens, f/5.6 deep crisp focus, natural lighting, lifelike skin pores and real-world material textures, authentic documentary cinematography, no cartoon, no illustration, no CGI, no text",
    negativePrompt:
      "cartoon, anime, manga, comic, illustration, vector art, flat design, 3d render, cgi, unreal engine, painting, drawing, sketch, cel shaded, plastic skin, watermark, text, logo, blurry, deformed",
  };
}

interface NarrativeVisualBeat {
  index: number;
  spokenText: string;
  focalKeyword: string;
  photoQuery: string;
  imagePrompt: string;
}

interface AlignedNarrativeBeat extends NarrativeVisualBeat {
  startSec: number;
  endSec: number;
  durationSec: number;
}

function parseKeywordListInput(rawKeywords: string): string[] {
  const trimmed = (rawKeywords || "").trim();
  if (!trimmed) return [];
  if (trimmed.includes("|")) {
    return trimmed
      .split("|")
      .map((s) => s.trim())
      .filter(Boolean);
  }
  if (trimmed.includes("\n")) {
    return trimmed
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  }
  if (trimmed.includes(";")) {
    return trimmed
      .split(";")
      .map((s) => s.trim())
      .filter(Boolean);
  }
  const fullPromptSplit = trimmed
    .split(
      /,\s*(?=(?:Live-action\b|RAW\s+photograph\b|35mm\s+|50mm\s+|Wide\s+35mm\b|Close-up\s+|Intimate\s+50mm\b))/i
    )
    .map((s) => s.trim())
    .filter(Boolean);
  if (fullPromptSplit.length > 1) {
    return fullPromptSplit;
  }
  return trimmed
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function splitLongClauseAtNaturalBreak(clause: string): string[] {
  const clean = clause.trim();
  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length < 10) return [clean];

  const transitionRegex =
    /(?:,\s+(?=(?:when|while|where|until|before|after|as|then|suddenly|and\s+suddenly|revealing|but|yet|so|though|although|which|who|whom)\b)|;\s+|\s+—\s+|\s+-\s+|\s+(?=and\s+(?:suddenly|then|dives|leaps|runs|walks|steps|opens|sees|spots|finds|meets|turns|falls|rises|strikes|crashes|bursts|emerges|appears|slithers|jumps|climbs|enters|reaches|grabs|pulls|draws|flies)\b))/i;

  const match = transitionRegex.exec(clean);
  if (match && match.index > 0) {
    const cutPos = match.index + match[0].length;
    const left = clean.slice(0, cutPos).trim();
    const right = clean.slice(cutPos).trim();
    const leftWords = left.split(/\s+/).filter(Boolean).length;
    const rightWords = right.split(/\s+/).filter(Boolean).length;
    if (leftWords >= 4 && rightWords >= 4) {
      return [
        ...splitLongClauseAtNaturalBreak(left),
        ...splitLongClauseAtNaturalBreak(right),
      ];
    }
  }

  if (words.length > 18) {
    const midChar = Math.floor(clean.length / 2);
    const commaPositions: number[] = [];
    for (let i = 0; i < clean.length; i++) {
      if (clean[i] === "," && i + 1 < clean.length && clean[i + 1] === " ") {
        commaPositions.push(i + 1);
      }
    }
    let bestComma = -1;
    let bestDist = Infinity;
    for (const pos of commaPositions) {
      const lw = clean.slice(0, pos).trim().split(/\s+/).length;
      const rw = clean.slice(pos).trim().split(/\s+/).length;
      if (lw >= 5 && rw >= 5 && Math.abs(pos - midChar) < bestDist) {
        bestDist = Math.abs(pos - midChar);
        bestComma = pos;
      }
    }
    if (bestComma !== -1) {
      return [
        clean.slice(0, bestComma).trim(),
        clean.slice(bestComma).trim(),
      ];
    }

    const midWord = Math.floor(words.length / 2);
    let splitWordIdx = midWord;
    for (let offset = 0; offset <= 4; offset++) {
      for (const candidate of [midWord + offset, midWord - offset]) {
        if (
          candidate >= 6 &&
          words.length - candidate >= 6 &&
          /^(?:and|but|into|across|through|under|over|toward|towards|inside|outside|beneath|along|near|with|from|before|after|until|while|when|where)$/i.test(
            words[candidate]
          )
        ) {
          splitWordIdx = candidate;
          offset = 99;
          break;
        }
      }
    }
    return [
      words.slice(0, splitWordIdx).join(" "),
      words.slice(splitWordIdx).join(" "),
    ];
  }

  return [clean];
}

function segmentScriptIntoSpokenBeats(
  script: string,
  maxBeats = 10
): string[] {
  const normalized = script
    .replace(/\[(pause|silence):\s*[\d.]+s?\]/gi, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalized) return [];

  const rawSentences = normalized
    .split(/(?<=[.!?。！？])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const initialBeats: string[] = [];
  for (const sentence of rawSentences) {
    const subClauses = splitLongClauseAtNaturalBreak(sentence);
    for (const sc of subClauses) {
      if (sc) initialBeats.push(sc);
    }
  }

  const mergedShort: string[] = [];
  for (let i = 0; i < initialBeats.length; i++) {
    const cur = initialBeats[i];
    const wc = cur.split(/\s+/).filter(Boolean).length;
    if (wc < 3 && i + 1 < initialBeats.length) {
      initialBeats[i + 1] = `${cur} ${initialBeats[i + 1]}`.trim();
    } else if (wc < 3 && mergedShort.length > 0) {
      mergedShort[mergedShort.length - 1] =
        `${mergedShort[mergedShort.length - 1]} ${cur}`.trim();
    } else {
      mergedShort.push(cur);
    }
  }

  const targetCap = Math.max(2, Math.min(14, maxBeats));
  while (mergedShort.length > targetCap) {
    let bestIdx = 0;
    let bestCombinedWords = Infinity;
    for (let i = 0; i < mergedShort.length - 1; i++) {
      const combinedWords =
        mergedShort[i].split(/\s+/).length +
        mergedShort[i + 1].split(/\s+/).length;
      const penalty = /[.!?。！？]$/.test(mergedShort[i]) ? 6 : 0;
      if (combinedWords + penalty < bestCombinedWords) {
        bestCombinedWords = combinedWords + penalty;
        bestIdx = i;
      }
    }
    mergedShort.splice(
      bestIdx,
      2,
      `${mergedShort[bestIdx]} ${mergedShort[bestIdx + 1]}`.trim()
    );
  }

  return mergedShort;
}

const PRIORITY_VISUAL_ENTITIES = new Set([
  "snake",
  "snakes",
  "cobra",
  "viper",
  "python",
  "anaconda",
  "rattlesnake",
  "serpent",
  "lizard",
  "crocodile",
  "alligator",
  "turtle",
  "frog",
  "shark",
  "whale",
  "dolphin",
  "octopus",
  "jellyfish",
  "fish",
  "eagle",
  "hawk",
  "owl",
  "falcon",
  "crow",
  "raven",
  "parrot",
  "bird",
  "birds",
  "lion",
  "tiger",
  "leopard",
  "panther",
  "jaguar",
  "cheetah",
  "wolf",
  "wolves",
  "bear",
  "fox",
  "deer",
  "elephant",
  "rhino",
  "gorilla",
  "monkey",
  "horse",
  "horses",
  "dog",
  "cat",
  "rabbit",
  "spider",
  "scorpion",
  "bee",
  "bees",
  "butterfly",
  "ant",
  "ants",
  "dragon",
  "dinosaur",
  "waterfall",
  "volcano",
  "lava",
  "lightning",
  "thunderstorm",
  "storm",
  "tornado",
  "hurricane",
  "glacier",
  "iceberg",
  "ocean",
  "waves",
  "river",
  "stream",
  "lake",
  "pond",
  "cave",
  "cavern",
  "canyon",
  "desert",
  "dunes",
  "mountain",
  "mountains",
  "cliff",
  "forest",
  "jungle",
  "rainforest",
  "woods",
  "tree",
  "trees",
  "branch",
  "flower",
  "flowers",
  "rose",
  "fire",
  "flames",
  "campfire",
  "smoke",
  "snow",
  "blizzard",
  "rain",
  "moon",
  "sun",
  "sunrise",
  "sunset",
  "stars",
  "galaxy",
  "planet",
  "earth",
  "meteor",
  "asteroid",
  "sword",
  "blade",
  "dagger",
  "knife",
  "shield",
  "armor",
  "bow",
  "arrow",
  "compass",
  "clock",
  "watch",
  "hourglass",
  "key",
  "lock",
  "chest",
  "treasure",
  "gold",
  "coin",
  "coins",
  "diamond",
  "gem",
  "crystal",
  "crown",
  "ring",
  "book",
  "scroll",
  "letter",
  "mirror",
  "lantern",
  "torch",
  "candle",
  "lamp",
  "bridge",
  "castle",
  "palace",
  "temple",
  "ruins",
  "pyramid",
  "cabin",
  "cottage",
  "house",
  "tower",
  "lighthouse",
  "gate",
  "door",
  "stairs",
  "path",
  "trail",
  "road",
  "street",
  "city",
  "village",
  "market",
  "train",
  "locomotive",
  "ship",
  "boat",
  "canoe",
  "sailboat",
  "submarine",
  "airplane",
  "helicopter",
  "rocket",
  "spaceship",
  "car",
  "truck",
  "motorcycle",
  "bicycle",
  "microscope",
  "telescope",
  "camera",
  "robot",
  "engine",
  "coffee",
  "espresso",
  "bread",
  "tea",
  "apple",
  "fruit",
  "heart",
  "brain",
  "eye",
  "eyes",
  "hand",
  "hands",
  "explorer",
  "traveler",
  "hiker",
  "climber",
  "diver",
  "astronaut",
  "scientist",
  "doctor",
  "surgeon",
  "chef",
  "warrior",
  "knight",
  "king",
  "queen",
  "child",
  "boy",
  "girl",
  "farmer",
  "fisherman",
  "soldier",
]);

const BEAT_STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "that",
  "this",
  "with",
  "from",
  "into",
  "onto",
  "over",
  "under",
  "across",
  "through",
  "along",
  "around",
  "behind",
  "beneath",
  "beside",
  "between",
  "inside",
  "outside",
  "without",
  "within",
  "about",
  "above",
  "below",
  "after",
  "before",
  "until",
  "while",
  "where",
  "when",
  "what",
  "which",
  "who",
  "whom",
  "whose",
  "why",
  "how",
  "there",
  "here",
  "their",
  "they",
  "them",
  "then",
  "than",
  "these",
  "those",
  "your",
  "yours",
  "ours",
  "his",
  "her",
  "hers",
  "its",
  "our",
  "she",
  "him",
  "you",
  "are",
  "was",
  "were",
  "been",
  "being",
  "have",
  "has",
  "had",
  "having",
  "does",
  "did",
  "doing",
  "will",
  "would",
  "shall",
  "should",
  "can",
  "could",
  "may",
  "might",
  "must",
  "just",
  "only",
  "even",
  "still",
  "also",
  "very",
  "more",
  "most",
  "some",
  "many",
  "much",
  "such",
  "each",
  "every",
  "other",
  "another",
  "same",
  "both",
  "few",
  "own",
  "out",
  "off",
  "down",
  "away",
  "back",
  "right",
  "left",
  "front",
  "side",
  "part",
  "way",
  "time",
  "times",
  "moment",
  "moments",
  "second",
  "seconds",
  "minute",
  "minutes",
  "hour",
  "hours",
  "day",
  "days",
  "year",
  "years",
  "later",
  "suddenly",
  "quickly",
  "slowly",
  "quietly",
  "silently",
  "finally",
  "nearly",
  "almost",
  "always",
  "never",
  "ever",
  "maybe",
  "perhaps",
  "however",
  "instead",
  "though",
  "although",
  "because",
  "since",
  "something",
  "nothing",
  "everything",
  "someone",
  "anyone",
  "everyone",
  "nobody",
  "thing",
  "things",
  "fact",
  "truth",
  "story",
  "storyteller",
  "narrator",
  "says",
  "said",
  "tells",
  "told",
  "makes",
  "made",
  "gets",
  "got",
  "goes",
  "went",
  "comes",
  "came",
  "takes",
  "took",
  "sees",
  "saw",
  "looks",
  "looked",
  "knows",
  "knew",
  "thinks",
  "thought",
  "starts",
  "started",
  "begins",
  "began",
  "turns",
  "turned",
  "happens",
  "happened",
  "becomes",
  "became",
  "feels",
  "felt",
  "seems",
  "seemed",
  "keeps",
  "kept",
  "lets",
  "let",
  "like",
]);

const DESCRIPTIVE_ADJECTIVES = new Set([
  "green",
  "red",
  "blue",
  "black",
  "white",
  "golden",
  "silver",
  "dark",
  "bright",
  "glowing",
  "giant",
  "massive",
  "huge",
  "tiny",
  "small",
  "wild",
  "venomous",
  "poisonous",
  "deadly",
  "king",
  "ancient",
  "old",
  "young",
  "lone",
  "hidden",
  "secret",
  "wooden",
  "stone",
  "iron",
  "steel",
  "crystal",
  "glass",
  "muddy",
  "misty",
  "foggy",
  "rainy",
  "snowy",
  "frozen",
  "icy",
  "boiling",
  "burning",
  "blazing",
  "rushing",
  "roaring",
  "raging",
  "calm",
  "deep",
  "steep",
  "rocky",
  "mossy",
  "fallen",
  "broken",
  "emerald",
  "crimson",
  "tropical",
  "dense",
]);

function singularizeSimple(word: string): string {
  const lower = word.toLowerCase();
  if (lower === "wolves") return "wolf";
  if (lower === "leaves") return "leaf";
  if (lower === "knives") return "knife";
  if (lower.endsWith("ies") && lower.length > 4) {
    return `${lower.slice(0, -3)}y`;
  }
  if (
    lower.endsWith("s") &&
    !lower.endsWith("ss") &&
    !lower.endsWith("us") &&
    !lower.endsWith("is") &&
    lower.length > 3
  ) {
    return lower.slice(0, -1);
  }
  return lower;
}

function extractLiteralEntityFromClause(spokenText: string): {
  focalKeyword: string;
  phrase: string;
  isHighConfidenceEntity: boolean;
} {
  const words = spokenText
    .replace(/[^a-zA-Z0-9\s'-]/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^['-]+|['-]+$/g, ""))
    .filter((w) => w.length >= 2);

  const creaturePriority = new Set([
    "snake",
    "snakes",
    "cobra",
    "viper",
    "python",
    "anaconda",
    "rattlesnake",
    "serpent",
    "lizard",
    "crocodile",
    "alligator",
    "turtle",
    "frog",
    "shark",
    "whale",
    "dolphin",
    "octopus",
    "jellyfish",
    "eagle",
    "hawk",
    "owl",
    "falcon",
    "crow",
    "raven",
    "parrot",
    "lion",
    "tiger",
    "leopard",
    "panther",
    "jaguar",
    "cheetah",
    "wolf",
    "wolves",
    "bear",
    "fox",
    "deer",
    "elephant",
    "rhino",
    "gorilla",
    "monkey",
    "horse",
    "dog",
    "cat",
    "rabbit",
    "spider",
    "scorpion",
    "butterfly",
    "dragon",
    "dinosaur",
  ]);

  let matchedIdx = -1;
  for (let i = 0; i < words.length; i++) {
    const low = words[i].toLowerCase();
    const sing = singularizeSimple(low);
    if (creaturePriority.has(low) || creaturePriority.has(sing)) {
      matchedIdx = i;
      break;
    }
  }

  if (matchedIdx === -1) {
    for (let i = 0; i < words.length; i++) {
      const low = words[i].toLowerCase();
      const sing = singularizeSimple(low);
      if (
        PRIORITY_VISUAL_ENTITIES.has(low) ||
        PRIORITY_VISUAL_ENTITIES.has(sing)
      ) {
        matchedIdx = i;
        break;
      }
    }
  }

  if (matchedIdx !== -1) {
    const headNoun = singularizeSimple(words[matchedIdx]);
    const modifiers: string[] = [];
    for (let j = Math.max(0, matchedIdx - 2); j < matchedIdx; j++) {
      const prevLow = words[j].toLowerCase();
      if (DESCRIPTIVE_ADJECTIVES.has(prevLow)) {
        modifiers.push(prevLow);
      }
    }
    const phrase = [...modifiers, headNoun].slice(-3).join(" ");
    return {
      focalKeyword: headNoun,
      phrase: phrase || headNoun,
      isHighConfidenceEntity: true,
    };
  }

  const detSet = new Set(["a", "an", "the", "this", "that", "one", "two"]);
  for (let i = 0; i < words.length - 1; i++) {
    if (detSet.has(words[i].toLowerCase())) {
      const npTokens: string[] = [];
      for (let j = i + 1; j < Math.min(words.length, i + 4); j++) {
        const tok = words[j].toLowerCase();
        if (BEAT_STOPWORDS.has(tok) || detSet.has(tok)) break;
        if (tok.length >= 3) npTokens.push(tok);
      }
      if (npTokens.length > 0) {
        const head = singularizeSimple(npTokens[npTokens.length - 1]);
        if (!BEAT_STOPWORDS.has(head) && head.length >= 3) {
          return {
            focalKeyword: head,
            phrase: [...npTokens.slice(0, -1), head].slice(-3).join(" "),
            isHighConfidenceEntity: false,
          };
        }
      }
    }
  }

  const contentWords = words
    .map((w) => w.toLowerCase())
    .filter((w) => w.length >= 3 && !BEAT_STOPWORDS.has(w));
  if (contentWords.length > 0) {
    const focal = singularizeSimple(contentWords[0]);
    return {
      focalKeyword: focal,
      phrase: contentWords.slice(0, 3).join(" "),
      isHighConfidenceEntity: false,
    };
  }

  return {
    focalKeyword: "",
    phrase: "",
    isHighConfidenceEntity: false,
  };
}

function textMentionsKeywordStem(text: string, keyword: string): boolean {
  const kw = singularizeSimple(keyword.trim());
  if (!kw || kw.length < 3) return true;
  const escaped = kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped}(?:s|es|ing|ed)?\\b`, "i").test(text);
}

function enforceBeatLiteralMatch(params: {
  rawBeat?: {
    focalKeyword?: string;
    photoQuery?: string;
    imagePrompt?: string;
  };
  spokenText: string;
  subject: string;
  shotIndex: number;
  enforcePhoto: boolean;
}): NarrativeVisualBeat {
  const framing =
    SHOT_FRAMING_VARIATIONS[params.shotIndex % SHOT_FRAMING_VARIATIONS.length];
  const literal = extractLiteralEntityFromClause(params.spokenText);
  const cleanSpokenClause = params.spokenText
    .replace(/["'`]/g, "")
    .replace(/\s+/g, " ")
    .trim();

  let focalKeyword = (params.rawBeat?.focalKeyword || "").trim().toLowerCase();
  let photoQuery = (params.rawBeat?.photoQuery || "").trim();
  let imagePrompt = (params.rawBeat?.imagePrompt || "").trim();

  imagePrompt = imagePrompt.replace(/^[^—]{2,45}\s+—\s+/u, "").trim();

  if (literal.isHighConfidenceEntity && literal.focalKeyword) {
    if (!textMentionsKeywordStem(imagePrompt, literal.focalKeyword)) {
      const worldContext =
        params.subject.trim() &&
        !textMentionsKeywordStem(params.subject, literal.focalKeyword)
          ? `, set within the context of ${params.subject.trim().slice(0, 50)}`
          : "";
      imagePrompt = params.enforcePhoto
        ? `Live-action 35mm RAW photograph of ${literal.phrase} (${cleanSpokenClause})${worldContext}, ${framing}`
        : `${literal.phrase}, illustrating "${cleanSpokenClause}"${worldContext}, ${framing}`;
    }
    if (!textMentionsKeywordStem(photoQuery, literal.focalKeyword)) {
      photoQuery = literal.phrase || literal.focalKeyword;
    }
    focalKeyword = literal.focalKeyword;
  } else {
    if (!focalKeyword) {
      focalKeyword =
        literal.focalKeyword ||
        extractLiteralEntityFromClause(params.subject).focalKeyword ||
        "scene";
    }
    if (!photoQuery) {
      photoQuery =
        literal.phrase ||
        focalKeyword ||
        params.subject.trim().split(/\s+/).slice(0, 3).join(" ");
    }
    if (!imagePrompt) {
      const coreDesc =
        literal.phrase || cleanSpokenClause || params.subject.trim();
      imagePrompt = params.enforcePhoto
        ? `Live-action 35mm RAW photograph of ${coreDesc}, illustrating ${cleanSpokenClause}, ${framing}`
        : `${coreDesc}, illustrating ${cleanSpokenClause}, ${framing}`;
    }
  }

  return {
    index: params.shotIndex,
    spokenText: params.spokenText,
    focalKeyword: singularizeSimple(focalKeyword.split(/\s+/)[0] || focalKeyword),
    photoQuery: photoQuery.slice(0, 60),
    imagePrompt,
  };
}

async function buildBeatVisualPlan(params: {
  subject: string;
  script: string;
  customRequirements?: string;
  maxBeats?: number;
  userKeywordHints?: string[];
}): Promise<NarrativeVisualBeat[]> {
  const spokenBeats = segmentScriptIntoSpokenBeats(
    params.script,
    params.maxBeats ?? 10
  );
  const styleDesc = buildPhotorealisticStyleDescriptor(
    params.subject,
    params.customRequirements || ""
  );

  if (spokenBeats.length === 0) {
    const basis = params.subject.trim() || "documentary scene";
    return [0, 1, 2, 3].map((idx) =>
      enforceBeatLiteralMatch({
        spokenText: basis,
        subject: basis,
        shotIndex: idx,
        enforcePhoto: styleDesc.enforcePhoto,
      })
    );
  }

  const numberedBeats = spokenBeats
    .map((text, idx) => `Beat ${idx}: "${text}"`)
    .join("\n");

  const hintsBlock =
    params.userKeywordHints && params.userKeywordHints.length > 0
      ? `Optional Visual Context Hints: ${params.userKeywordHints
          .slice(0, 8)
          .join(" | ")}`
      : "";

  const prompt = [
    `You are synchronizing video clips 1-to-1 with a storyteller's spoken voiceover beats.`,
    `There are EXACTLY ${spokenBeats.length} spoken beats below (Beat 0 through Beat ${
      spokenBeats.length - 1
    }). Return a JSON object with a "beats" array containing EXACTLY ${
      spokenBeats.length
    } items in the exact same order.`,
    params.subject.trim()
      ? `Overall Story Setting / Topic Context: ${params.subject.trim()}`
      : "",
    params.customRequirements?.trim()
      ? `Custom Visual Requirements: ${params.customRequirements.trim()}`
      : "",
    hintsBlock,
    `SPOKEN VOICEOVER BEATS (IN EXACT CHRONOLOGICAL ORDER):\n${numberedBeats}`,
    `MANDATORY LITERAL VOICE-TO-CLIP MATCHING RULES:
1. Every Beat i MUST depict the EXACT concrete subject, creature, animal, person, action, object, or place spoken in Beat i!
   - Example: If Beat i says "when suddenly a venomous snake slithers across a branch", Beat i MUST show a real snake slithering on a branch ("focalKeyword": "snake", "photoQuery": "green viper snake branch"). NEVER show a generic landscape or overall topic shot when a specific noun is spoken!
2. "focalKeyword": 1 concrete English singular noun naming the primary physical subject spoken in Beat i (e.g., "snake", "waterfall", "explorer", "compass", "volcano").
3. "photoQuery": 2 to 3 concrete English photographic search nouns matching Beat i (e.g., "green tree snake", "jungle muddy trail", "rushing waterfall pool").
4. "imagePrompt": Self-contained 15-to-25 word English ${
      styleDesc.enforcePhoto
        ? "live-action 35mm RAW camera photograph description (real textures, natural lighting, no cartoon, no CGI, no text)"
        : "scene description preserving requested art style (no text overlays)"
    } starting directly with the concrete subject of Beat i.`,
  ]
    .filter(Boolean)
    .join("\n\n");

  let rawAiBeats: Array<{
    focalKeyword?: string;
    photoQuery?: string;
    imagePrompt?: string;
  }> = [];

  try {
    const ai = getGenAIClient();
    const response = await generateTextWithFallback(ai, {
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            beats: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  focalKeyword: { type: Type.STRING },
                  photoQuery: { type: Type.STRING },
                  imagePrompt: { type: Type.STRING },
                },
                required: ["focalKeyword", "photoQuery", "imagePrompt"],
              },
            },
          },
          required: ["beats"],
        },
      },
    });
    const parsed = parseJsonObjectSafely<{
      beats?: Array<{
        focalKeyword?: string;
        photoQuery?: string;
        imagePrompt?: string;
      }>;
    }>(response.text ?? "{}");
    if (Array.isArray(parsed?.beats) && parsed.beats.length > 0) {
      rawAiBeats = parsed.beats;
    }
  } catch (err) {
    if (!isTransientDemandError(err) && !(err instanceof DOMException)) {
      throw err;
    }
  }

  if (rawAiBeats.length === 0) {
    try {
      const rawBackup = await callPollinationsTextFallback(
        `Return STRICTLY a valid JSON object: {"beats": [{"focalKeyword": "snake", "photoQuery": "green snake branch", "imagePrompt": "Live-action 35mm RAW photograph of ..."}, ...]} with EXACTLY ${spokenBeats.length} items matching Beat 0..${
          spokenBeats.length - 1
        }. No markdown.`,
        prompt
      );
      const backupParsed = parseJsonObjectSafely<{
        beats?: Array<{
          focalKeyword?: string;
          photoQuery?: string;
          imagePrompt?: string;
        }>;
      }>(rawBackup);
      if (Array.isArray(backupParsed?.beats) && backupParsed.beats.length > 0) {
        rawAiBeats = backupParsed.beats;
      }
    } catch {
      // Fall back cleanly to deterministic per-beat literal entity extraction below
    }
  }

  return spokenBeats.map((spokenText, idx) =>
    enforceBeatLiteralMatch({
      rawBeat: rawAiBeats[idx],
      spokenText,
      subject: params.subject,
      shotIndex: idx,
      enforcePhoto: styleDesc.enforcePhoto,
    })
  );
}

async function generateScriptAndKeywordsViaGemini(params: {
  subject: string;
  language: string;
  paragraphCount: number;
  customRequirements: string;
  systemPrompt: string;
}): Promise<{
  script: string;
  keywords: string[];
  photoQueries?: string[];
}> {
  const langInstruction =
    params.language && params.language !== "Auto Detect"
      ? `Write the spoken narration script in ${params.language}.`
      : "Write the spoken narration script in the same language as the Video Subject.";

  const targetParagraphs = Math.max(
    1,
    Math.min(10, Number(params.paragraphCount) || 1)
  );

  const styleDesc = buildPhotorealisticStyleDescriptor(
    params.subject,
    params.customRequirements
  );

  const realismRule = styleDesc.enforcePhoto
    ? `3. PHOTOREALISM RULE (MANDATORY): Every single item in "keywords" MUST describe an authentic LIVE-ACTION REAL-WORLD PHOTOGRAPH shot with a real 35mm/50mm camera (real people, real animals, real places, real physical objects, natural lighting, realistic textures). NEVER use cartoon, anime, illustration, vector, comic, 3D render, or abstract CGI metaphors.`
    : `3. Every single item in "keywords" MUST explicitly name the specific subject/character/object/location from the user's prompt AND preserve any requested visual style. Never include text/words inside the image.`;

  const promptParts = [
    `USER VIDEO PROMPT / SUBJECT (FOLLOW THIS STRICTLY): ${params.subject}`,
    `Target Paragraph Count: ${targetParagraphs}`,
    langInstruction,
    params.customRequirements?.trim()
      ? `Mandatory Custom Requirements & Style Instructions: ${params.customRequirements.trim()}`
      : "",
    `CRITICAL INSTRUCTIONS FOR "script", "keywords", AND "photoQueries":
1. The "script" MUST strictly and directly fulfill the exact topic, story, facts, and tone requested in USER VIDEO PROMPT / SUBJECT and any Mandatory Custom Requirements. Do NOT drift into generic commentary.
2. The "keywords" array MUST contain sequential, self-contained English live-action camera scene descriptions (14 to 24 words each) ordered 1-to-1 with each sentence/clause beat of the narration script so when the narrator mentions a specific creature, object, or action (e.g., "a snake"), that exact beat's visual prompt shows that literal subject. Never include text overlays.
${realismRule}
4. The "photoQueries" array MUST contain concise real-world English photographic search phrases (2 to 3 concrete nouns each) matching each scene in "keywords" one-to-one.`,
  ]
    .filter(Boolean)
    .join("\n\n");

  const baseSystemPrompt =
    params.systemPrompt?.trim() ||
    "You are an expert documentary & cinema video scriptwriter and director of photography. Strictly follow the user's exact prompt with authentic live-action photorealistic cinematography. Return raw spoken narration only without markdown titles, narrator labels, or stage directions.";

  let generatedScript = "";

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
                "Raw spoken narration script strictly about the user's exact Video Subject, divided into the requested number of paragraphs without titles or narrator labels.",
            },
            keywords: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            photoQueries: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ["script", "keywords"],
        },
      },
    });

    const parsed = parseJsonObjectSafely<{
      script?: string;
      keywords?: string[];
      photoQueries?: string[];
    }>(response.text ?? "{}");

    if (parsed?.script?.trim()) {
      generatedScript = parsed.script.trim();
      const beats = await buildBeatVisualPlan({
        subject: params.subject,
        script: generatedScript,
        customRequirements: params.customRequirements,
        maxBeats: 10,
        userKeywordHints: parsed.keywords,
      });
      return {
        script: generatedScript,
        keywords: beats.map((b) => b.imagePrompt),
        photoQueries: beats.map((b) => b.photoQuery),
      };
    }
  } catch (err) {
    if (!isTransientDemandError(err) && !(err instanceof DOMException)) {
      throw err;
    }
  }

  const rawBackup = await callPollinationsTextFallback(
    `${baseSystemPrompt}\n\nRespond STRICTLY as a valid JSON object with keys: "script" (raw spoken narration strictly matching the user's prompt), "keywords" (array of detailed English live-action 35mm photograph descriptions matching each sentence beat), and "photoQueries" (array of concise 2-to-3 word real photo search queries). Do not wrap in markdown.`,
    promptParts
  );

  const backupParsed = parseJsonObjectSafely<{
    script?: string;
    keywords?: string[];
    photoQueries?: string[];
  }>(rawBackup);

  generatedScript = (backupParsed?.script || rawBackup || "").trim();
  const alignedBeats = await buildBeatVisualPlan({
    subject: params.subject,
    script: generatedScript,
    customRequirements: params.customRequirements,
    maxBeats: 10,
    userKeywordHints: backupParsed?.keywords,
  });

  return {
    script: generatedScript,
    keywords: alignedBeats.map((b) => b.imagePrompt),
    photoQueries: alignedBeats.map((b) => b.photoQuery),
  };
}

async function extractKeywordsWithFailover(
  subject: string,
  script: string,
  customRequirements = ""
): Promise<{
  keywords: string[];
  photoQueries?: string[];
  beats?: NarrativeVisualBeat[];
}> {
  if (script.trim()) {
    const beats = await buildBeatVisualPlan({
      subject,
      script,
      customRequirements,
      maxBeats: 10,
    });
    return {
      keywords: beats.map((b) => b.imagePrompt),
      photoQueries: beats.map((b) => b.photoQuery),
      beats,
    };
  }

  const styleDesc = buildPhotorealisticStyleDescriptor(
    subject,
    customRequirements
  );
  const basis = subject.trim() || "documentary scene";
  const fallbackBeats = [0, 1, 2, 3].map((idx) =>
    enforceBeatLiteralMatch({
      spokenText: basis,
      subject: basis,
      shotIndex: idx,
      enforcePhoto: styleDesc.enforcePhoto,
    })
  );
  return {
    keywords: fallbackBeats.map((b) => b.imagePrompt),
    photoQueries: fallbackBeats.map((b) => b.photoQuery),
    beats: fallbackBeats,
  };
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
  entranceAnimation?: string;
  uppercase?: boolean;
  fontColor: string;
  fontSize: number;
  strokeColor: string;
  strokeWidth: number;
  shadowEnabled?: boolean;
  shadowColor?: string;
  shadowOffset?: number;
  shadowBlur?: number;
  shadowOpacity?: number;
  bgEnabled: boolean;
  bgColor: string;
  alignedBeats?: AlignedNarrativeBeat[];
}): string {
  interface TimedCaptionChunk {
    text: string;
    start: number;
    end: number;
  }

  const useUppercase = params.uppercase !== false;
  const maxSentenceChars = useUppercase ? 56 : 72;
  const wordsPerSentenceWrap = useUppercase ? 7 : 9;

  const timedChunks: TimedCaptionChunk[] = [];

  if (params.alignedBeats && params.alignedBeats.length > 0) {
    for (const beat of params.alignedBeats) {
      const beatText = beat.spokenText.trim();
      if (!beatText) continue;
      const beatSpan = Math.max(0.25, beat.endSec - beat.startSec);

      let subParts: string[] = [];
      if (params.displayMode === "Word by word") {
        const words = beatText.split(/\s+/).filter(Boolean);
        for (let i = 0; i < words.length; i += 3) {
          subParts.push(words.slice(i, i + 3).join(" "));
        }
      } else if (beatText.length > maxSentenceChars) {
        const words = beatText.split(/\s+/).filter(Boolean);
        for (let i = 0; i < words.length; i += wordsPerSentenceWrap) {
          subParts.push(words.slice(i, i + wordsPerSentenceWrap).join(" "));
        }
      } else {
        subParts = [beatText];
      }

      const subTotalChars = Math.max(
        1,
        subParts.reduce((acc, s) => acc + s.length, 0)
      );
      let subCursor = beat.startSec;
      for (let i = 0; i < subParts.length; i++) {
        const dur =
          i === subParts.length - 1
            ? Math.max(0.15, beat.endSec - subCursor)
            : Math.max(0.2, (subParts[i].length / subTotalChars) * beatSpan);
        const start = subCursor;
        const end = Math.min(beat.endSec, subCursor + dur);
        subCursor = end;
        timedChunks.push({ text: subParts[i], start, end });
      }
    }
  } else {
    const chunks: string[] = [];
    if (params.displayMode === "Word by word") {
      const words = params.script
        .split(/\s+/)
        .map((w) => w.trim())
        .filter(Boolean);
      for (let i = 0; i < words.length; i += 3) {
        chunks.push(words.slice(i, i + 3).join(" "));
      }
    } else {
      const rawSentences = params.script
        .split(/(?<=[.!?。！？,;；])\s+/)
        .map((s) => s.trim())
        .filter(Boolean);
      for (const s of rawSentences) {
        if (s.length > maxSentenceChars) {
          const words = s.split(/\s+/);
          for (let i = 0; i < words.length; i += wordsPerSentenceWrap) {
            chunks.push(words.slice(i, i + wordsPerSentenceWrap).join(" "));
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
    let cursor = 0;
    chunks.forEach((chunk, idx) => {
      const segDur =
        idx === chunks.length - 1
          ? Math.max(0.4, params.duration - cursor)
          : Math.max(0.5, (chunk.length / totalChars) * params.duration);
      const start = cursor;
      const end = Math.min(params.duration, cursor + segDur);
      cursor = end;
      timedChunks.push({ text: chunk, start, end });
    });
  }

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
    const pct = Math.max(
      5,
      Math.min(95, Number(params.customPositionPercent) || 80)
    );
    alignment = 8;
    marginV = Math.round((pct / 100) * params.height);
  }

  const shadowEnabled = params.shadowEnabled !== false;
  const clampedShadowOpacity = Math.max(
    0,
    Math.min(100, Number(params.shadowOpacity ?? 80))
  );
  const shadowAlphaByte = Math.round(
    ((100 - clampedShadowOpacity) / 100) * 255
  );
  const shadowAlphaHex = shadowAlphaByte
    .toString(16)
    .padStart(2, "0")
    .toUpperCase();
  const shadowAssCol = shadowEnabled
    ? hexToAssColor(params.shadowColor || "#000000", shadowAlphaHex)
    : hexToAssColor("#000000", "FF");
  const shadowPx = shadowEnabled
    ? Math.max(0.5, Math.min(16, (Number(params.shadowOffset ?? 4) || 4) * 1.15))
    : 0;
  const shadowBlurPx = shadowEnabled
    ? Math.max(0, Math.min(10, Number(params.shadowBlur ?? 2)))
    : 0;

  const primaryCol = hexToAssColor(params.fontColor, "00");
  const outlineCol = params.bgEnabled
    ? hexToAssColor(params.bgColor, "40")
    : hexToAssColor(params.strokeColor, "00");
  const borderStyle = params.bgEnabled ? 3 : 1;
  const rawStrokeNum = Number(params.strokeWidth ?? 1.5);
  const outlinePx = params.bgEnabled
    ? 8
    : Math.max(0, Math.min(14, (Number.isFinite(rawStrokeNum) ? rawStrokeNum : 1.5) * 1.4));
  const scaledFontPx = Math.max(
    24,
    Math.min(74, Math.round((Number(params.fontSize) || 60) * 0.72))
  );

  // Use a dedicated blurred shadow layer on Layer 0 when soft shadow blur is active (keeps main text & outline razor-sharp)
  const useSeparateBlurShadowLayer =
    shadowEnabled && !params.bgEnabled && shadowBlurPx > 0;
  const defaultStyleShadowPx = useSeparateBlurShadowLayer ? 0 : shadowPx;
  const defaultBackCol = useSeparateBlurShadowLayer
    ? hexToAssColor("#000000", "FF")
    : shadowAssCol;
  const transparentAssCol = hexToAssColor("#000000", "FF");

  const styleLines = [
    `Style: Default,DejaVu Sans,${scaledFontPx},${primaryCol},${primaryCol},${outlineCol},${defaultBackCol},1,0,0,0,100,100,0,0,${borderStyle},${outlinePx.toFixed(
      1
    )},${defaultStyleShadowPx.toFixed(1)},${alignment},48,48,${marginV},1`,
  ];
  if (useSeparateBlurShadowLayer) {
    styleLines.push(
      `Style: TextShadow,DejaVu Sans,${scaledFontPx},${transparentAssCol},${transparentAssCol},${transparentAssCol},${shadowAssCol},1,0,0,0,100,100,0,0,1,${outlinePx.toFixed(
        1
      )},${shadowPx.toFixed(1)},${alignment},48,48,${marginV},1`
    );
  }

  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: ${params.width}
PlayResY: ${params.height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
${styleLines.join("\n")}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`;

  let kineticBodyTags = "";
  if (params.entranceAnimation === "Spring pop") {
    kineticBodyTags =
      "\\fad(65,65)\\fscx84\\fscy84\\t(0,135,\\fscx105\\fscy105)\\t(135,230,\\fscx100\\fscy100)";
  } else if (params.entranceAnimation === "Kinetic Motion Ease") {
    kineticBodyTags =
      "\\fad(90,80)\\fscx94\\fscy94\\t(0,160,\\fscx100\\fscy100)";
  } else if (params.entranceAnimation === "Subtle Ease") {
    kineticBodyTags =
      "\\fad(120,100)\\fscx98\\fscy98\\t(0,150,\\fscx100\\fscy100)";
  } else if (
    !params.entranceAnimation ||
    params.entranceAnimation === "Cinema Clean Dissolve"
  ) {
    kineticBodyTags = "\\fad(140,115)";
  }

  const mainPrefix = kineticBodyTags ? `{${kineticBodyTags}}` : "";
  const shadowPrefix = `{${kineticBodyTags}\\blur${shadowBlurPx.toFixed(1)}}`;

  const events: string[] = [];
  for (const item of timedChunks) {
    const cleaned = item.text.replace(/[{}\\]/g, "").replace(/\r?\n/g, " ");
    const safeText = useUppercase ? cleaned.toUpperCase() : cleaned;
    const startTs = formatAssTimestamp(item.start);
    const endTs = formatAssTimestamp(item.end);

    if (useSeparateBlurShadowLayer) {
      events.push(
        `Dialogue: 0,${startTs},${endTs},TextShadow,,0,0,0,,${shadowPrefix}${safeText}`
      );
      events.push(
        `Dialogue: 1,${startTs},${endTs},Default,,0,0,0,,${mainPrefix}${safeText}`
      );
    } else {
      events.push(
        `Dialogue: 0,${startTs},${endTs},Default,,0,0,0,,${mainPrefix}${safeText}`
      );
    }
  }

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

interface DetectedSilenceInterval {
  start: number;
  end: number;
  duration: number;
  cutPoint: number;
}

function countSpokenSyllables(text: string): number {
  const cleaned = text.trim();
  if (!cleaned) return 1;

  const nonAsciiChars = (cleaned.match(/[^\x00-\x7F]/g) || []).length;
  const asciiWords = cleaned
    .replace(/[^\x00-\x7F]/g, " ")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  let syllables = nonAsciiChars * 0.85;
  for (const w of asciiWords) {
    if (/^\d+$/.test(w)) {
      syllables += Math.max(1, w.length * 1.2);
      continue;
    }
    const core = w.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "");
    const groups = core.match(/[aeiouy]{1,2}/g);
    syllables += Math.max(1, groups ? groups.length : 1);
  }

  return Math.max(1.5, syllables);
}

async function detectAudioSilences(
  wavPath: string,
  totalDuration: number
): Promise<DetectedSilenceInterval[]> {
  try {
    const { stderr } = await execFileAsync(
      "ffmpeg",
      [
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        wavPath,
        "-af",
        "silencedetect=noise=-35dB:d=0.065",
        "-f",
        "null",
        "-",
      ],
      { timeout: 15_000, maxBuffer: 4 * 1024 * 1024 }
    );

    const intervals: DetectedSilenceInterval[] = [];
    let pendingStart: number | null = null;
    for (const line of stderr.split(/\r?\n/)) {
      const startMatch = line.match(/silence_start:\s*([\d.]+)/);
      if (startMatch) {
        pendingStart = parseFloat(startMatch[1]);
        continue;
      }
      const endMatch = line.match(/silence_end:\s*([\d.]+)/);
      if (endMatch && pendingStart !== null) {
        const end = Math.min(totalDuration, parseFloat(endMatch[1]));
        const start = Math.max(0, Math.min(end, pendingStart));
        const duration = Math.max(0, end - start);
        if (duration >= 0.05) {
          const cutPoint = start + duration * 0.58;
          intervals.push({ start, end, duration, cutPoint });
        }
        pendingStart = null;
      }
    }

    if (pendingStart !== null && pendingStart < totalDuration - 0.04) {
      const start = Math.max(0, pendingStart);
      const duration = totalDuration - start;
      intervals.push({
        start,
        end: totalDuration,
        duration,
        cutPoint: start + duration * 0.5,
      });
    }

    return intervals;
  } catch {
    return [];
  }
}

async function alignBeatsToVoiceAudio(
  beats: NarrativeVisualBeat[],
  totalDuration: number,
  wavPath?: string
): Promise<AlignedNarrativeBeat[]> {
  const n = beats.length;
  if (n === 0) return [];
  if (n === 1) {
    return [
      {
        ...beats[0],
        startSec: 0,
        endSec: totalDuration,
        durationSec: totalDuration,
      },
    ];
  }

  const silences =
    wavPath && fs.existsSync(wavPath)
      ? await detectAudioSilences(wavPath, totalDuration)
      : [];

  let speechStart = 0;
  let speechEnd = totalDuration;
  if (silences.length > 0) {
    const first = silences[0];
    if (first.start <= 0.05 && first.end < Math.min(0.85, totalDuration * 0.14)) {
      speechStart = first.end;
    }
    const last = silences[silences.length - 1];
    if (
      last.end >= totalDuration - 0.06 &&
      last.start > Math.max(totalDuration - 1.2, totalDuration * 0.84)
    ) {
      speechEnd = last.start;
    }
  }

  const activeSpan = Math.max(1.0, speechEnd - speechStart);
  const weights = beats.map((b, idx) => {
    const syl = countSpokenSyllables(b.spokenText);
    const trimmed = b.spokenText.trim();
    let pauseBonus = 0;
    if (idx < n - 1) {
      if (/[.!?。！？]+$/.test(trimmed)) {
        pauseBonus = 1.85;
      } else if (/[,;—:，；]+$/.test(trimmed)) {
        pauseBonus = 0.85;
      }
    }
    const internalPauses =
      (trimmed.slice(0, -1).match(/[,;—:]/g) || []).length * 0.45;
    return Math.max(2.0, syl + pauseBonus + internalPauses);
  });

  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const minBeatSec = Math.min(0.95, Math.max(0.55, totalDuration / (n * 2.1)));
  const internalSilences = silences.filter(
    (s) => s.cutPoint > speechStart + 0.35 && s.cutPoint < speechEnd - 0.35
  );
  const usedSilenceIndices = new Set<number>();

  const cuts: number[] = [];
  let cumWeight = 0;

  for (let i = 0; i < n - 1; i++) {
    cumWeight += weights[i];
    const expectedCut = speechStart + (cumWeight / totalWeight) * activeSpan;
    const prevCut = i === 0 ? 0 : cuts[i - 1];
    const minAllowed = prevCut + minBeatSec;
    const maxAllowed = totalDuration - (n - 1 - i) * minBeatSec;

    const isSentenceEnd = /[.!?。！？]+$/.test(beats[i].spokenText.trim());
    const avgBeatSpan = activeSpan / n;
    const snapWindow = isSentenceEnd
      ? Math.min(1.65, Math.max(0.65, avgBeatSpan * 0.65))
      : Math.min(1.0, Math.max(0.45, avgBeatSpan * 0.45));

    let chosenCut = Math.max(minAllowed, Math.min(maxAllowed, expectedCut));
    let bestScore = Infinity;
    let bestSilenceIdx = -1;

    for (let sIdx = 0; sIdx < internalSilences.length; sIdx++) {
      if (usedSilenceIndices.has(sIdx)) continue;
      const s = internalSilences[sIdx];
      if (s.cutPoint < minAllowed || s.cutPoint > maxAllowed) continue;
      const dist = Math.abs(s.cutPoint - expectedCut);
      if (dist > snapWindow) continue;
      const score = dist - Math.min(0.38, s.duration * 0.85);
      if (score < bestScore) {
        bestScore = score;
        bestSilenceIdx = sIdx;
      }
    }

    if (bestSilenceIdx !== -1) {
      usedSilenceIndices.add(bestSilenceIdx);
      chosenCut = Math.max(
        minAllowed,
        Math.min(maxAllowed, internalSilences[bestSilenceIdx].cutPoint)
      );
    }

    cuts.push(chosenCut);
  }

  return beats.map((beat, idx) => {
    const startSec = idx === 0 ? 0 : cuts[idx - 1];
    const endSec = idx === n - 1 ? totalDuration : cuts[idx];
    return {
      ...beat,
      startSec,
      endSec,
      durationSec: Math.max(0.4, endSec - startSec),
    };
  });
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

const SHOT_FRAMING_VARIATIONS = [
  "35mm wide establishing documentary photograph, natural ambient light",
  "50mm medium-close photographic detail, realistic surface textures",
  "35mm candid medium camera composition, authentic environment",
  "low-angle 35mm architectural and environmental perspective",
  "wide 35mm cinema still frame, crisp depth and natural lighting",
  "intimate 50mm prime lens focal portrait or object detail",
  "atmospheric wide-angle live-action documentary photograph",
];

function buildStoryboardPrompts(params: {
  keywords: string[];
  subject: string;
  customRequirements?: string;
  targetCount: number;
  concatMode?: string;
  matchToScript?: boolean;
}): string[] {
  const base = params.keywords.map((k) => k.trim()).filter(Boolean);
  const subjectClean = params.subject.trim();
  const expanded: string[] = [];

  for (let i = 0; i < params.targetCount; i++) {
    const framing = SHOT_FRAMING_VARIATIONS[i % SHOT_FRAMING_VARIATIONS.length];
    if (i < base.length) {
      const kw = base[i];
      const anchored =
        kw.split(/\s+/).length < 5 &&
        subjectClean &&
        !kw.toLowerCase().includes(subjectClean.toLowerCase().slice(0, 12))
          ? `${subjectClean} — ${kw}, ${framing}`
          : kw;
      expanded.push(anchored);
    } else if (base.length > 0) {
      const seedPrompt = base[i % base.length];
      expanded.push(`${seedPrompt}, ${framing}`);
    } else {
      expanded.push(`${subjectClean}, ${framing}`);
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
      lastPollinationsVideoNotice =
        "Animated 35mm Cinema Clips (Zoom, Pan & Slide)";
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

let pollinationsImageCooldownUntil = 0;
let pollinationsPublicNextSlotMs = 0;
let pollinationsPublicQueuePromise: Promise<void> = Promise.resolve();
const gradioSpaceCooldownUntil = new Map<string, number>();

const WIKIMEDIA_USER_AGENT =
  "MoneyPrinterStudio/1.0 (https://localhost:3000; cinema-storyboard-engine) node-fetch";

let commonsQueuePromise: Promise<void> = Promise.resolve();

function runPolitelySpaced<T>(task: () => Promise<T>, delayMs = 240): Promise<T> {
  const next = commonsQueuePromise.then(async () => {
    const out = await task();
    await sleep(delayMs);
    return out;
  });
  commonsQueuePromise = next.then(
    () => undefined,
    () => undefined
  );
  return next;
}

function runPollinationsPublicSpaced<T>(
  task: () => Promise<T>,
  delayMs = 1350
): Promise<T> {
  const next = pollinationsPublicQueuePromise.then(async () => {
    const out = await task();
    await sleep(delayMs);
    return out;
  });
  pollinationsPublicQueuePromise = next.then(
    () => undefined,
    () => undefined
  );
  return next;
}

function getSafeDiffusionDims(
  width: number,
  height: number
): { gWidth: number; gHeight: number } {
  if (height > width) {
    return { gWidth: 576, gHeight: 1024 };
  }
  if (width > height) {
    return { gWidth: 1024, gHeight: 576 };
  }
  return { gWidth: 1024, gHeight: 1024 };
}

async function masterPhotoBufferToCinemaJpg(params: {
  buf: Buffer;
  workDir: string;
  tag: string;
  destPath: string;
  targetWidth: number;
  targetHeight: number;
  trimBottomWatermark?: boolean;
}): Promise<boolean> {
  if (!params.buf || params.buf.length < 4096) {
    return false;
  }

  const rawTempPath = path.join(
    params.workDir,
    `raw_photo_${params.tag}_${Date.now()}_${Math.floor(Math.random() * 9999)}.img`
  );

  try {
    fs.writeFileSync(rawTempPath, params.buf);
    const maxDim = Math.max(params.targetWidth, params.targetHeight);
    const cropPrefix = params.trimBottomWatermark
      ? "crop=iw:trunc(ih*0.952/2)*2:0:0,"
      : "";
    const vf = `${cropPrefix}scale='min(${maxDim},iw)':'min(${maxDim},ih)':force_original_aspect_ratio=decrease:flags=lanczos,scale=trunc(iw/2)*2:trunc(ih/2)*2,unsharp=5:5:0.35:5:5:0.0`;

    await execFileAsync(
      "ffmpeg",
      [
        "-y",
        "-loglevel",
        "error",
        "-i",
        rawTempPath,
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        params.destPath,
      ],
      FFMPEG_EXEC_OPTS
    );

    return (
      fs.existsSync(params.destPath) && fs.statSync(params.destPath).size > 8192
    );
  } catch {
    return false;
  } finally {
    try {
      if (fs.existsSync(rawTempPath)) fs.unlinkSync(rawTempPath);
    } catch {
      // ignore temp cleanup
    }
  }
}

async function generateViaGradioImageSpace(params: {
  baseUrl: string;
  endpoint: string;
  payload: unknown[];
  workDir: string;
  shotIndex: number;
  targetWidth: number;
  targetHeight: number;
  destPath: string;
}): Promise<boolean> {
  const cooldown = gradioSpaceCooldownUntil.get(params.baseUrl) || 0;
  if (Date.now() < cooldown) {
    return false;
  }

  try {
    const callRes = await fetch(
      `${params.baseUrl}/gradio_api/call${params.endpoint}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: params.payload }),
        signal: AbortSignal.timeout(10_000),
      }
    );

    if (callRes.status === 429 || callRes.status === 503) {
      gradioSpaceCooldownUntil.set(params.baseUrl, Date.now() + 180_000);
      return false;
    }
    if (!callRes.ok) {
      gradioSpaceCooldownUntil.set(params.baseUrl, Date.now() + 90_000);
      return false;
    }

    const callJson = (await callRes.json()) as { event_id?: string };
    const eventId = callJson?.event_id;
    if (!eventId) {
      return false;
    }

    const sseRes = await fetch(
      `${params.baseUrl}/gradio_api/call${params.endpoint}/${eventId}`,
      {
        signal: AbortSignal.timeout(32_000),
      }
    );
    if (!sseRes.ok) {
      return false;
    }

    const sseText = await sseRes.text();
    if (sseText.includes("event: error")) {
      // Immediately cool down ZeroGPU space when per-IP GPU quota is depleted
      gradioSpaceCooldownUntil.set(params.baseUrl, Date.now() + 180_000);
      return false;
    }

    const lines = sseText.split(/\r?\n/);
    let completeDataLine = "";
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim() === "event: complete") {
        for (let j = i + 1; j < lines.length; j++) {
          if (lines[j].startsWith("data:")) {
            completeDataLine = lines[j].slice(5).trim();
            break;
          }
        }
        if (completeDataLine) break;
      }
    }

    if (!completeDataLine) {
      return false;
    }

    const parsed = JSON.parse(completeDataLine) as Array<{
      url?: string;
      path?: string;
    }>;
    const firstItem = Array.isArray(parsed) ? parsed[0] : null;
    const fileUrl =
      firstItem?.url ||
      (firstItem?.path
        ? `${params.baseUrl}/gradio_api/file=${firstItem.path}`
        : "");

    if (!fileUrl) {
      return false;
    }

    const imgRes = await fetch(fileUrl, {
      signal: AbortSignal.timeout(15_000),
    });
    if (!imgRes.ok) {
      return false;
    }

    const imgBuf = Buffer.from(await imgRes.arrayBuffer());
    return await masterPhotoBufferToCinemaJpg({
      buf: imgBuf,
      workDir: params.workDir,
      tag: `gradio_${params.shotIndex}`,
      destPath: params.destPath,
      targetWidth: params.targetWidth,
      targetHeight: params.targetHeight,
    });
  } catch {
    return false;
  }
}

const CAMERA_JARGON_STOPWORDS = new Set([
  "live",
  "action",
  "live-action",
  "raw",
  "photograph",
  "photography",
  "photo",
  "shot",
  "on",
  "with",
  "from",
  "the",
  "and",
  "for",
  "into",
  "over",
  "under",
  "through",
  "during",
  "near",
  "arri",
  "alexa",
  "leica",
  "sony",
  "dslr",
  "35mm",
  "50mm",
  "85mm",
  "prime",
  "lens",
  "wide",
  "wide-angle",
  "establishing",
  "documentary",
  "editorial",
  "candid",
  "medium",
  "medium-close",
  "close-up",
  "closeup",
  "intimate",
  "low-angle",
  "angle",
  "perspective",
  "composition",
  "view",
  "frame",
  "still",
  "cinema",
  "cinematic",
  "natural",
  "ambient",
  "daylight",
  "lighting",
  "light",
  "realistic",
  "photorealistic",
  "lifelike",
  "authentic",
  "crisp",
  "sharp",
  "focus",
  "detail",
  "detailed",
  "surface",
  "textures",
  "texture",
  "environment",
  "environmental",
  "atmospheric",
  "depth",
  "dynamic",
  "dramatic",
  "vivid",
  "scene",
  "how",
  "why",
  "what",
  "when",
  "where",
  "changing",
  "everyday",
  "about",
  "inside",
  "outside",
  "into",
  "onto",
  "within",
  "across",
  "through",
  "around",
  "behind",
  "under",
  "over",
  "close",
  "closeup",
  "macro",
  "glowing",
  "rustic",
  "soft",
  "warm",
  "dark",
  "thick",
  "golden",
  "polished",
  "freshly",
  "professional",
  "artisan",
  "uses",
  "using",
  "monitors",
  "watching",
  "pulls",
  "falling",
  "rising",
  "streaming",
]);

const NON_PHOTO_TITLE_REGEX =
  /\.(svg|gif|tif|tiff|pdf|djvu|xcf|ogg|webm|stl)$|(\b(icon|logo|symbol|diagram|chart|plot|graph|map|locator|flag|coat_of_arms|blason|crest|seal|emblem|cartoon|comic|anime|manga|illustration|drawing|sketch|clipart|vector|signature|autograph|stamp|coin|banknote|medal|page|scan|document|manuscript|newspaper|poster|flyer|screenshot|ui|interface|blank|placeholder|wordcloud|statue|monument|memorial)\b)/i;

function buildRealPhotoSearchCandidates(
  rawPrompt: string,
  subject: string,
  explicitPhotoQuery?: string,
  secondaryPhotoQuery?: string,
  focalKeyword?: string
): string[] {
  const candidates: string[] = [];
  const pushUnique = (q: string) => {
    const trimmed = q.trim();
    if (
      trimmed &&
      !candidates.some((c) => c.toLowerCase() === trimmed.toLowerCase())
    ) {
      candidates.push(trimmed);
    }
  };

  const cleanTokens = (text: string, maxWords: number): string => {
    const words = text
      .replace(/[^a-zA-Z0-9\s-]/g, " ")
      .split(/\s+/)
      .map((w) => w.trim())
      .filter(
        (w) =>
          w.length >= 3 && !CAMERA_JARGON_STOPWORDS.has(w.toLowerCase())
      );
    const dedup: string[] = [];
    for (const w of words) {
      if (!dedup.some((d) => d.toLowerCase() === w.toLowerCase())) {
        dedup.push(w);
      }
      if (dedup.length >= maxWords) break;
    }
    return dedup.join(" ");
  };

  const cleanFocal = (focalKeyword || "").trim().toLowerCase();

  if (explicitPhotoQuery?.trim()) {
    pushUnique(cleanTokens(explicitPhotoQuery, 3));
  }

  if (cleanFocal && cleanFocal !== "scene") {
    pushUnique(cleanFocal);
  }

  pushUnique(cleanTokens(rawPrompt, 3));

  if (secondaryPhotoQuery?.trim()) {
    pushUnique(cleanTokens(secondaryPhotoQuery, 3));
  }

  if (!cleanFocal || cleanFocal === "scene") {
    const coreSubjectClause =
      subject.split(/\b(?:in|at|near|from)\b/i)[0] || subject;
    pushUnique(cleanTokens(coreSubjectClause, 4));
    pushUnique(cleanTokens(subject, 5));
  }

  if (candidates.length === 0 && subject.trim()) {
    pushUnique(subject.trim().slice(0, 40));
  }

  return candidates;
}

async function tryFetchRealDocumentaryPhoto(params: {
  searchCandidates: string[];
  focalKeyword?: string;
  shotIndex: number;
  targetWidth: number;
  targetHeight: number;
  workDir: string;
  destPath: string;
  usedUrls: Set<string>;
}): Promise<boolean> {
  const cleanFocal = (params.focalKeyword || "").trim().toLowerCase();
  const hasSpecificFocal = Boolean(
    cleanFocal && cleanFocal.length >= 3 && cleanFocal !== "scene"
  );

  // 1. Wikimedia Commons High-Resolution Bitmap Photography Search
  for (const query of params.searchCandidates) {
    if (!query.trim()) continue;
    try {
      const apiUrl = `https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(
        `filetype:bitmap ${query}`
      )}&gsrnamespace=6&gsrlimit=20&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=1600&format=json`;

      const pages = await runPolitelySpaced(async () => {
        const res = await fetch(apiUrl, {
          headers: { "User-Agent": WIKIMEDIA_USER_AGENT },
          signal: AbortSignal.timeout(10_000),
        });
        if (!res.ok) return [];
        const json = (await res.json()) as {
          query?: {
            pages?: Record<
              string,
              {
                title?: string;
                imageinfo?: Array<{
                  url?: string;
                  thumburl?: string;
                  mime?: string;
                  width?: number;
                  height?: number;
                  size?: number;
                }>;
              }
            >;
          };
        };
        return Object.values(json?.query?.pages || {});
      }, 220);

      const validPhotoItems = pages
        .filter((p) => {
          const title = p.title || "";
          const ii = p.imageinfo?.[0];
          if (!ii) return false;
          if (NON_PHOTO_TITLE_REGEX.test(title)) return false;
          const mime = (ii.mime || "").toLowerCase();
          if (!/^image\/(jpeg|jpg|webp|png)$/.test(mime)) return false;
          if ((ii.width || 0) < 640 || (ii.height || 0) < 420) return false;
          if ((ii.size || 0) < 22_000) return false;
          const dlUrl = ii.thumburl || ii.url || "";
          if (!dlUrl || params.usedUrls.has(dlUrl)) return false;
          return true;
        })
        .sort((a, b) => {
          const aFocal =
            hasSpecificFocal && textMentionsKeywordStem(a.title || "", cleanFocal)
              ? 0
              : 1;
          const bFocal =
            hasSpecificFocal && textMentionsKeywordStem(b.title || "", cleanFocal)
              ? 0
              : 1;
          if (aFocal !== bFocal) return aFocal - bFocal;
          const aJpg = (a.imageinfo?.[0]?.mime || "").includes("jpeg") ? 0 : 1;
          const bJpg = (b.imageinfo?.[0]?.mime || "").includes("jpeg") ? 0 : 1;
          return aJpg - bJpg;
        });

      const filteredForFocal = hasSpecificFocal
        ? validPhotoItems.filter((p) =>
            textMentionsKeywordStem(p.title || "", cleanFocal)
          )
        : validPhotoItems;

      const itemsToTry =
        filteredForFocal.length > 0
          ? filteredForFocal
          : query.toLowerCase() === cleanFocal
          ? validPhotoItems
          : [];

      for (const candidate of itemsToTry.slice(0, 8)) {
        const ii = candidate.imageinfo![0];
        const dlUrl = ii.thumburl || ii.url!;
        params.usedUrls.add(dlUrl);
        try {
          const imgRes = await fetch(dlUrl, {
            headers: { "User-Agent": WIKIMEDIA_USER_AGENT },
            signal: AbortSignal.timeout(12_000),
          });
          if (!imgRes.ok) continue;
          const buf = Buffer.from(await imgRes.arrayBuffer());
          const ok = await masterPhotoBufferToCinemaJpg({
            buf,
            workDir: params.workDir,
            tag: `commons_${params.shotIndex}`,
            destPath: params.destPath,
            targetWidth: params.targetWidth,
            targetHeight: params.targetHeight,
          });
          if (ok) return true;
        } catch {
          // try next photo result
        }
      }
    } catch {
      // try next search query
    }
  }

  // 2. Wikipedia High-Res PageImages Real Photography Search
  for (const query of params.searchCandidates.slice(0, 3)) {
    if (!query.trim()) continue;
    try {
      const wikiUrl = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(
        query
      )}&gsrlimit=8&prop=pageimages&piprop=original|thumbnail&pithumbsize=1600&format=json`;

      const pages = await runPolitelySpaced(async () => {
        const res = await fetch(wikiUrl, {
          headers: { "User-Agent": WIKIMEDIA_USER_AGENT },
          signal: AbortSignal.timeout(9_000),
        });
        if (!res.ok) return [];
        const json = (await res.json()) as {
          query?: {
            pages?: Record<
              string,
              {
                title?: string;
                original?: { source?: string; width?: number; height?: number };
                thumbnail?: { source?: string; width?: number; height?: number };
              }
            >;
          };
        };
        return Object.values(json?.query?.pages || {});
      }, 200);

      const orderedPages = [...pages].sort((a, b) => {
        const aMatch =
          hasSpecificFocal &&
          textMentionsKeywordStem(
            `${a.title || ""} ${a.original?.source || ""}`,
            cleanFocal
          )
            ? 0
            : 1;
        const bMatch =
          hasSpecificFocal &&
          textMentionsKeywordStem(
            `${b.title || ""} ${b.original?.source || ""}`,
            cleanFocal
          )
            ? 0
            : 1;
        return aMatch - bMatch;
      });

      for (const p of orderedPages) {
        const src = p.thumbnail?.source || p.original?.source || "";
        if (!src || params.usedUrls.has(src)) continue;
        if (
          NON_PHOTO_TITLE_REGEX.test(src) ||
          NON_PHOTO_TITLE_REGEX.test(p.title || "")
        ) {
          continue;
        }
        if (
          hasSpecificFocal &&
          query.toLowerCase() !== cleanFocal &&
          !textMentionsKeywordStem(`${p.title || ""} ${src}`, cleanFocal)
        ) {
          continue;
        }
        params.usedUrls.add(src);
        try {
          const imgRes = await fetch(src, {
            headers: { "User-Agent": WIKIMEDIA_USER_AGENT },
            signal: AbortSignal.timeout(12_000),
          });
          if (!imgRes.ok) continue;
          const buf = Buffer.from(await imgRes.arrayBuffer());
          const ok = await masterPhotoBufferToCinemaJpg({
            buf,
            workDir: params.workDir,
            tag: `wiki_${params.shotIndex}`,
            destPath: params.destPath,
            targetWidth: params.targetWidth,
            targetHeight: params.targetHeight,
          });
          if (ok) return true;
        } catch {
          // try next page image
        }
      }
    } catch {
      // fall through to Flickr DSLR feed
    }
  }

  // 3. Flickr Public Real DSLR Photography Feed
  const flickrTagSets: string[][] = [];
  for (const query of params.searchCandidates.slice(0, 2)) {
    const tags = query
      .split(/\s+/)
      .map((t) => t.replace(/[^a-zA-Z0-9]/g, "").toLowerCase())
      .filter((t) => t.length >= 3)
      .slice(0, 2);
    if (tags.length > 0) flickrTagSets.push(tags);
  }
  if (hasSpecificFocal) {
    flickrTagSets.push([cleanFocal]);
  }

  for (const tags of flickrTagSets) {
    try {
      const flickrUrl = `https://www.flickr.com/services/feeds/photos_public.gne?tags=${encodeURIComponent(
        tags.join(",")
      )}&tagmode=all&format=json&nojsoncallback=1`;
      const res = await fetch(flickrUrl, {
        headers: { "User-Agent": "Mozilla/5.0 Chrome/131.0.0.0" },
        signal: AbortSignal.timeout(9_000),
      });
      if (!res.ok) continue;
      const json = (await res.json()) as {
        items?: Array<{ title?: string; tags?: string; media?: { m?: string } }>;
      };
      const items = json?.items || [];
      for (const item of items) {
        if (NON_PHOTO_TITLE_REGEX.test(item.title || "")) continue;
        if (
          hasSpecificFocal &&
          !textMentionsKeywordStem(
            `${item.title || ""} ${item.tags || ""}`,
            cleanFocal
          )
        ) {
          continue;
        }
        const rawM = item.media?.m || "";
        if (!rawM) continue;
        const hiResUrl = rawM.replace(/_m\.(jpg|jpeg|png)$/i, "_b.$1");
        if (params.usedUrls.has(hiResUrl)) continue;
        params.usedUrls.add(hiResUrl);
        try {
          const imgRes = await fetch(hiResUrl, {
            signal: AbortSignal.timeout(10_000),
          });
          if (!imgRes.ok) continue;
          const buf = Buffer.from(await imgRes.arrayBuffer());
          const ok = await masterPhotoBufferToCinemaJpg({
            buf,
            workDir: params.workDir,
            tag: `flickr_${params.shotIndex}`,
            destPath: params.destPath,
            targetWidth: params.targetWidth,
            targetHeight: params.targetHeight,
          });
          if (ok) return true;
        } catch {
          // try next Flickr DSLR item
        }
      }
    } catch {
      // continue
    }
  }

  // 4. High-Resolution Photographic Plate Safety Net (100% real camera photography, zero SVG/cartoon)
  try {
    const picUrl = `https://picsum.photos/seed/cinema_${
      params.shotIndex * 97 + Date.now() % 1000
    }/${params.targetWidth}/${params.targetHeight}`;
    const picRes = await fetch(picUrl, { signal: AbortSignal.timeout(10_000) });
    if (picRes.ok) {
      const buf = Buffer.from(await picRes.arrayBuffer());
      const ok = await masterPhotoBufferToCinemaJpg({
        buf,
        workDir: params.workDir,
        tag: `plate_${params.shotIndex}`,
        destPath: params.destPath,
        targetWidth: params.targetWidth,
        targetHeight: params.targetHeight,
      });
      if (ok) return true;
    }
  } catch {
    // fall through to bundled local photographs
  }

  const localPhotos = getFallbackLocalImages();
  if (localPhotos.length > 0) {
    const srcPhoto = localPhotos[params.shotIndex % localPhotos.length];
    try {
      const buf = fs.readFileSync(srcPhoto);
      return await masterPhotoBufferToCinemaJpg({
        buf,
        workDir: params.workDir,
        tag: `local_${params.shotIndex}`,
        destPath: params.destPath,
        targetWidth: params.targetWidth,
        targetHeight: params.targetHeight,
      });
    } catch {
      return false;
    }
  }

  return false;
}

async function fetchPollinationsStoryboardFrames(params: {
  prompts: string[];
  photoQueries?: string[];
  focalKeywords?: string[];
  subject: string;
  customRequirements?: string;
  width: number;
  height: number;
  workDir: string;
  onSceneProgress?: (completed: number, total: number) => void;
}): Promise<string[]> {
  const savedPaths: string[] = new Array(params.prompts.length).fill("");
  const apiKey = process.env.POLLINATIONS_API_KEY;
  let completedCount = 0;
  const totalCount = params.prompts.length;
  const { gWidth, gHeight } = getSafeDiffusionDims(params.width, params.height);
  const usedRealPhotoUrls = new Set<string>();

  const styleDesc = buildPhotorealisticStyleDescriptor(
    params.subject,
    params.customRequirements
  );

  await mapWithConcurrency(
    params.prompts,
    Math.min(2, Math.max(1, totalCount)),
    async (rawPrompt, idx) => {
      if (idx > 0) {
        await sleep(idx * 180);
      }
      const destPath = path.join(params.workDir, `scene_${idx}.jpg`);
      const enriched = `${rawPrompt}, ${styleDesc.cameraModifier}`;
      const seed = Math.floor(1000 + Math.random() * 900000) + idx * 173;
      const safeSeed = seed % 2147483000;

      // Tier 1: Authenticated Pollinations AI Flux (when Pollen balance is available)
      if (apiKey && Date.now() >= pollinationsImageCooldownUntil) {
        const pollinationsModels = ["flux", "flux-klein"];
        for (let mIdx = 0; mIdx < pollinationsModels.length; mIdx++) {
          const modelName = pollinationsModels[mIdx];
          const url = `https://gen.pollinations.ai/image/${encodeURIComponent(
            enriched
          )}?model=${modelName}&width=${params.width}&height=${
            params.height
          }&nologo=true&seed=${seed + mIdx * 17}`;
          try {
            const res = await fetch(url, {
              headers: { Authorization: `Bearer ${apiKey}` },
              signal: AbortSignal.timeout(28_000),
            });
            if (res.status === 401 || res.status === 402) {
              pollinationsImageCooldownUntil = Date.now() + 180_000;
              break;
            }
            if (res.status === 429 || res.status >= 500) {
              continue;
            }
            const ct = (res.headers.get("content-type") || "").toLowerCase();
            if (res.ok && ct.includes("image")) {
              const buf = Buffer.from(await res.arrayBuffer());
              const ok = await masterPhotoBufferToCinemaJpg({
                buf,
                workDir: params.workDir,
                tag: `polli_auth_${idx}`,
                destPath,
                targetWidth: params.width,
                targetHeight: params.height,
              });
              if (ok) {
                savedPaths[idx] = destPath;
                break;
              }
            }
          } catch {
            // Fall through to Tier 2
          }
        }
      }

      // Tier 2: Public Pollinations Flux Endpoint (spaced queue so multiple beats get bespoke AI scenes)
      if (!savedPaths[idx] && Date.now() >= pollinationsPublicNextSlotMs) {
        await runPollinationsPublicSpaced(async () => {
          if (Date.now() < pollinationsPublicNextSlotMs) return;
          try {
            const pubW = params.height > params.width ? 768 : 1280;
            const pubH = params.height > params.width ? 1344 : 720;
            const pubUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(
              enriched
            )}?width=${pubW}&height=${pubH}&model=flux&nologo=true&seed=${safeSeed}`;
            const res = await fetch(pubUrl, {
              headers: {
                "User-Agent":
                  "Mozilla/5.0 (X11; Linux x86_64) Chrome/131.0.0.0 Safari/537.36",
              },
              signal: AbortSignal.timeout(24_000),
            });
            if (res.status === 402 || res.status === 429 || res.status === 503) {
              pollinationsPublicNextSlotMs = Date.now() + 35_000;
            } else if (res.ok) {
              const ct = (res.headers.get("content-type") || "").toLowerCase();
              if (ct.includes("image")) {
                const buf = Buffer.from(await res.arrayBuffer());
                const ok = await masterPhotoBufferToCinemaJpg({
                  buf,
                  workDir: params.workDir,
                  tag: `polli_pub_${idx}`,
                  destPath,
                  targetWidth: params.width,
                  targetHeight: params.height,
                  trimBottomWatermark: true,
                });
                if (ok) {
                  savedPaths[idx] = destPath;
                }
              }
            }
          } catch {
            pollinationsPublicNextSlotMs = Date.now() + 15_000;
          }
        }, 1250);
      }

      // Tier 3: Direct Photorealistic FLUX.1-schnell / FLUX.1-merged / SDXL-Flash GPU Mirrors
      if (!savedPaths[idx]) {
        const fluxSchnellMirror = {
          baseUrl: "https://black-forest-labs-flux-1-schnell.hf.space",
          endpoint: "/infer",
          payload: [enriched, safeSeed, false, gWidth, gHeight, 4],
        };
        const fluxMergedMirror = {
          baseUrl: "https://multimodalart-flux-1-merged.hf.space",
          endpoint: "/infer",
          payload: [enriched, safeSeed, false, gWidth, gHeight, 3.5, 8],
        };
        const sdxlFlashMirror = {
          baseUrl: "https://kingnish-sdxl-flash.hf.space",
          endpoint: "/run",
          payload: [
            enriched,
            styleDesc.negativePrompt,
            true,
            safeSeed,
            gWidth,
            gHeight,
            3,
            8,
            false,
          ],
        };

        const orderedMirrors =
          idx % 2 === 0
            ? [fluxSchnellMirror, fluxMergedMirror, sdxlFlashMirror]
            : [fluxMergedMirror, fluxSchnellMirror, sdxlFlashMirror];

        for (const mirror of orderedMirrors) {
          const ok = await generateViaGradioImageSpace({
            baseUrl: mirror.baseUrl,
            endpoint: mirror.endpoint,
            payload: mirror.payload,
            workDir: params.workDir,
            shotIndex: idx,
            targetWidth: params.width,
            targetHeight: params.height,
            destPath,
          });
          if (ok) {
            savedPaths[idx] = destPath;
            break;
          }
        }
      }

      // Tier 4 & 5: Focal-Verified Real Documentary Photography
      if (!savedPaths[idx]) {
        const pqList = params.photoQueries || [];
        const fkList = params.focalKeywords || [];
        const primaryPhotoQuery = idx < pqList.length ? pqList[idx] : undefined;
        const beatFocalKeyword = idx < fkList.length ? fkList[idx] : undefined;
        const searchCandidates = buildRealPhotoSearchCandidates(
          rawPrompt,
          params.subject,
          primaryPhotoQuery,
          undefined,
          beatFocalKeyword
        );
        const realPhotoOk = await tryFetchRealDocumentaryPhoto({
          searchCandidates,
          focalKeyword: beatFocalKeyword,
          shotIndex: idx,
          targetWidth: params.width,
          targetHeight: params.height,
          workDir: params.workDir,
          destPath,
          usedUrls: usedRealPhotoUrls,
        });
        if (realPhotoOk) {
          savedPaths[idx] = destPath;
        }
      }

      completedCount += 1;
      params.onSceneProgress?.(completedCount, totalCount);
    }
  );

  const validGenerated = savedPaths.filter(Boolean);
  if (validGenerated.length === 0) {
    throw new Error(
      "Photorealistic scene acquisition failed unexpectedly. Please click Generate Video again."
    );
  }

  for (let i = 0; i < savedPaths.length; i++) {
    if (!savedPaths[i]) {
      const donor = validGenerated[i % validGenerated.length];
      const cloneDest = path.join(params.workDir, `scene_${i}.jpg`);
      fs.copyFileSync(donor, cloneDest);
      savedPaths[i] = cloneDest;
    }
  }

  return savedPaths;
}

function resolveXfadeTransition(
  mode: string,
  index: number
): {
  transition: string;
  duration: number;
} {
  const baseDur = 0.42;

  const shufflePool = [
    "smoothleft",
    "fade",
    "smoothright",
    "circleopen",
    "slideleft",
    "fadeblack",
    "smoothup",
  ];
  const slideInPool = ["slideleft", "slideright", "slideup", "slidedown"];
  const slideOutPool = ["smoothleft", "smoothright", "smoothup", "smoothdown"];

  switch (mode) {
    case "None":
      return { transition: "none", duration: 0 };
    case "FadeIn":
      return { transition: "fade", duration: baseDur };
    case "FadeOut":
      return { transition: "fadeblack", duration: baseDur };
    case "SlideIn":
      return {
        transition: slideInPool[index % slideInPool.length],
        duration: baseDur,
      };
    case "SlideOut":
      return {
        transition: slideOutPool[index % slideOutPool.length],
        duration: baseDur,
      };
    case "ZoomIn":
      return { transition: "circleopen", duration: baseDur };
    case "ZoomOut":
      return { transition: "circleclose", duration: baseDur };
    case "Shuffle":
    default:
      return {
        transition: shufflePool[index % shufflePool.length],
        duration: baseDur,
      };
  }
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
      ? `scale=${params.width}:${params.height}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${params.width}:${params.height}:(ow-iw)/2:(oh-ih)/2:black`
      : `scale=${params.width}:${params.height}:force_original_aspect_ratio=increase:flags=lanczos,crop=${params.width}:${params.height}`;

  const targetDur = (params.clipDuration + 0.18).toFixed(3);
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
      `setpts=${ptsFactor}*PTS,${scaleExpr},eq=contrast=1.03:saturation=0.99,fps=${VIDEO_RENDER_FPS},setsar=1,format=yuv420p`,
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

type PhotoMotionPreset =
  | "zoom-in"
  | "zoom-out"
  | "slide-left"
  | "slide-right"
  | "slide-up"
  | "slide-down"
  | "diag-zoom-in"
  | "diag-zoom-out"
  | "static";

function resolvePhotoMotionPreset(params: {
  photoAnimationMode?: string;
  videoTransitionMode?: string;
  shotIndex: number;
}): PhotoMotionPreset {
  const rawAnim = (params.photoAnimationMode || "").trim();
  const idx = Math.max(0, params.shotIndex);

  if (rawAnim === "None (Static Photo)" || rawAnim === "None") {
    return "static";
  }

  switch (rawAnim) {
    case "Zoom In":
    case "ZoomIn":
      return "zoom-in";
    case "Zoom Out":
    case "ZoomOut":
      return "zoom-out";
    case "Alternate Zoom (In & Out)":
      return idx % 2 === 0 ? "zoom-in" : "zoom-out";
    case "Slide Left & Right":
      return idx % 2 === 0 ? "slide-right" : "slide-left";
    case "Slide Left":
      return "slide-left";
    case "Slide Right":
      return "slide-right";
    case "Slide Up & Down":
      return idx % 2 === 0 ? "slide-up" : "slide-down";
    case "Ken Burns Drift (Pan + Zoom)":
      return idx % 2 === 0 ? "diag-zoom-in" : "diag-zoom-out";
    default:
      break;
  }

  // If photoAnimationMode was omitted (e.g., direct API call), honor directional videoTransitionMode hints
  if (!rawAnim && params.videoTransitionMode) {
    if (params.videoTransitionMode === "ZoomIn") return "zoom-in";
    if (params.videoTransitionMode === "ZoomOut") return "zoom-out";
    if (
      params.videoTransitionMode === "SlideIn" ||
      params.videoTransitionMode === "SlideOut"
    ) {
      return idx % 2 === 0 ? "slide-right" : "slide-left";
    }
  }

  const dynamicCycle: PhotoMotionPreset[] = [
    "zoom-in",
    "slide-right",
    "zoom-out",
    "slide-left",
    "diag-zoom-in",
    "slide-up",
    "diag-zoom-out",
    "slide-down",
  ];
  return dynamicCycle[idx % dynamicCycle.length];
}

async function renderAnimatedSceneClipFromFrame(params: {
  imagePath: string;
  outputClipPath: string;
  width: number;
  height: number;
  clipDuration: number;
  clipSpeed: number;
  fitMode: string;
  photoAnimationMode?: string;
  videoTransitionMode?: string;
  shotIndex: number;
}): Promise<void> {
  const targetSec = Math.max(1.5, params.clipDuration + 0.18);
  const preset = resolvePhotoMotionPreset({
    photoAnimationMode: params.photoAnimationMode,
    videoTransitionMode: params.videoTransitionMode,
    shotIndex: params.shotIndex,
  });

  if (preset === "static") {
    const staticScaleExpr =
      params.fitMode === "Fit (black bars)"
        ? `scale=${params.width}:${params.height}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${params.width}:${params.height}:(ow-iw)/2:(oh-ih)/2:black`
        : `scale=${params.width}:${params.height}:force_original_aspect_ratio=increase:flags=lanczos,crop=${params.width}:${params.height}`;

    const staticVf = `${staticScaleExpr},unsharp=5:5:0.30:5:5:0.0,eq=contrast=1.02:saturation=0.99,fps=${VIDEO_RENDER_FPS},setsar=1,format=yuv420p`;

    await execFileAsync(
      "ffmpeg",
      [
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        String(VIDEO_RENDER_FPS),
        "-i",
        params.imagePath,
        "-t",
        targetSec.toFixed(3),
        "-vf",
        staticVf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        params.outputClipPath,
      ],
      FFMPEG_EXEC_OPTS
    );
    return;
  }

  // 3x Lanczos supersampling eliminates 1px integer stair-stepping in FFmpeg zoompan (0.33px subpixel precision)
  const preW = params.width * 3;
  const preH = params.height * 3;
  const preScaleExpr =
    params.fitMode === "Fit (black bars)"
      ? `scale=${preW}:${preH}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${preW}:${preH}:(ow-iw)/2:(oh-ih)/2:black`
      : `scale=${preW}:${preH}:force_original_aspect_ratio=increase:flags=lanczos,crop=${preW}:${preH}`;

  const totalFrames = Math.max(30, Math.ceil(targetSec * VIDEO_RENDER_FPS));
  const amp = Math.max(0.75, Math.min(1.45, params.clipSpeed || 1.0));

  // Smooth comma-free progress curve in [0, 1]: immediate gentle cruise velocity + cosine ease
  const p = `(0.65*(on/${totalFrames})+0.35*(0.5-0.5*cos(PI*on/${totalFrames})))`;

  let zExpr = "1.0";
  let xExpr = "iw/2-(iw/zoom/2)";
  let yExpr = "ih/2-(ih/zoom/2)";

  const zoomSpan = (0.24 * amp).toFixed(4);
  const slideBaseZoom = (1.18 + 0.04 * amp).toFixed(4);
  const slideDriftZoom = (0.04 * amp).toFixed(4);
  const diagSpan = (0.22 * amp).toFixed(4);
  const diagMax = (1.05 + 0.22 * amp).toFixed(4);

  switch (preset) {
    case "zoom-in":
      zExpr = `1.0+${zoomSpan}*${p}`;
      xExpr = "iw/2-(iw/zoom/2)";
      yExpr = "ih/2-(ih/zoom/2)";
      break;
    case "zoom-out":
      zExpr = `${(1.02 + 0.24 * amp).toFixed(4)}-${zoomSpan}*${p}`;
      xExpr = "iw/2-(iw/zoom/2)";
      yExpr = "ih/2-(ih/zoom/2)";
      break;
    case "slide-left":
      zExpr = `${slideBaseZoom}+${slideDriftZoom}*${p}`;
      xExpr = `(iw-iw/zoom)*(1-${p})`;
      yExpr = "(ih-ih/zoom)/2";
      break;
    case "slide-right":
      zExpr = `${slideBaseZoom}+${slideDriftZoom}*${p}`;
      xExpr = `(iw-iw/zoom)*${p}`;
      yExpr = "(ih-ih/zoom)/2";
      break;
    case "slide-up":
      zExpr = `${slideBaseZoom}+${slideDriftZoom}*${p}`;
      xExpr = "(iw-iw/zoom)/2";
      yExpr = `(ih-ih/zoom)*(1-${p})`;
      break;
    case "slide-down":
      zExpr = `${slideBaseZoom}+${slideDriftZoom}*${p}`;
      xExpr = "(iw-iw/zoom)/2";
      yExpr = `(ih-ih/zoom)*${p}`;
      break;
    case "diag-zoom-in":
      zExpr = `1.05+${diagSpan}*${p}`;
      xExpr = `(iw-iw/zoom)*(0.12+0.76*${p})`;
      yExpr = `(ih-ih/zoom)*(0.15+0.70*${p})`;
      break;
    case "diag-zoom-out":
      zExpr = `${diagMax}-${diagSpan}*${p}`;
      xExpr = `(iw-iw/zoom)*(0.88-0.76*${p})`;
      yExpr = `(ih-ih/zoom)*(0.85-0.70*${p})`;
      break;
  }

  const animatedVf = `${preScaleExpr},zoompan=z='${zExpr}':x='${xExpr}':y='${yExpr}':d=${totalFrames}:s=${params.width}x${params.height}:fps=${VIDEO_RENDER_FPS},unsharp=5:5:0.25:5:5:0.0,eq=contrast=1.02:saturation=0.99,setsar=1,format=yuv420p`;

  await execFileAsync(
    "ffmpeg",
    [
      "-y",
      "-loglevel",
      "error",
      "-i",
      params.imagePath,
      "-vf",
      animatedVf,
      "-frames:v",
      String(totalFrames),
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
      const {
        subject = "",
        script = "",
        customRequirements = "",
      } = req.body as {
        subject?: string;
        script?: string;
        customRequirements?: string;
      };

      const basis = (script || subject || "").trim();
      if (!basis) {
        res
          .status(400)
          .json({ error: "Please enter a Video Subject or Video Script first." });
        return;
      }

      const extracted = await extractKeywordsWithFailover(
        subject,
        script,
        customRequirements
      );
      res.json({ keywords: extracted.keywords });
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
    photoAnimationMode?: string;
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
    subtitleUppercase?: boolean;
    subtitlePosition?: string;
    customPositionPercent?: number;
    subtitleDisplayMode?: string;
    entranceAnimation?: string;
    fontColor?: string;
    fontSize?: number;
    strokeColor?: string;
    strokeWidth?: number;
    textShadowEnabled?: boolean;
    textShadowColor?: string;
    textShadowOffset?: number;
    textShadowBlur?: number;
    textShadowOpacity?: number;
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
      let keywordList = parseKeywordListInput(body.keywords || "");

      const effectiveSubject = rawSubject || finalScript.slice(0, 48);
      let aiPhotoQueries: string[] | undefined;
      let focalKeywords: string[] | undefined;
      const matchToScript = body.matchMaterialsToScript !== false;

      // 1. Generate Script via Gemini if missing
      if (!finalScript) {
        setJobProgress(
          jobId,
          8,
          "Generating script & beat-aligned 35mm visual scenes with Gemini AI..."
        );
        const gen = await generateScriptAndKeywordsViaGemini({
          subject: effectiveSubject,
          language: body.language || "en-US",
          paragraphCount: body.paragraphCount || 1,
          customRequirements: body.customRequirements || "",
          systemPrompt: body.systemPrompt || "",
        });
        if (gen.script) {
          finalScript = gen.script;
        }
        if (keywordList.length === 0 && gen.keywords.length > 0) {
          keywordList = gen.keywords;
        }
        if (gen.photoQueries && gen.photoQueries.length > 0) {
          aiPhotoQueries = gen.photoQueries;
        }
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

      // Estimate duration & compute spoken beat budget
      const voiceSpeedNum = parseFloat(body.voiceSpeed || "1.0") || 1.0;
      const estimatedSeconds = Math.max(
        6,
        Math.min(
          60,
          Math.round(finalScript.split(/\s+/).length / (2.4 * voiceSpeedNum))
        )
      );
      const clipSpeedNum = Math.max(
        0.5,
        Math.min(2.0, Number(body.clipSpeed) || 1.0)
      );
      const userMaxClipSec = Math.max(
        2.5,
        Math.min(6, Number(body.maxClipDuration) || 4)
      );
      const effectiveClipSec = Math.max(1.8, userMaxClipSec / clipSpeedNum);
      const maxBeatBudget = Math.max(
        4,
        Math.min(
          12,
          Math.ceil(estimatedSeconds / Math.max(2.0, effectiveClipSec * 0.8))
        )
      );

      let narrativeBeats: NarrativeVisualBeat[] | null = null;
      let shotPrompts: string[] = [];

      if (matchToScript && finalScript.trim()) {
        setJobProgress(
          jobId,
          11,
          "Mapping every spoken sentence & entity in your script to matching visual clips..."
        );
        narrativeBeats = await buildBeatVisualPlan({
          subject: effectiveSubject,
          script: finalScript,
          customRequirements: body.customRequirements || "",
          maxBeats: maxBeatBudget,
          userKeywordHints: keywordList,
        });
        shotPrompts = narrativeBeats.map((b) => b.imagePrompt);
        aiPhotoQueries = narrativeBeats.map((b) => b.photoQuery);
        focalKeywords = narrativeBeats.map((b) => b.focalKeyword);
        keywordList = shotPrompts;
      } else {
        if (keywordList.length === 0) {
          const extracted = await extractKeywordsWithFailover(
            effectiveSubject,
            finalScript,
            body.customRequirements || ""
          );
          keywordList = extracted.keywords;
          aiPhotoQueries = extracted.photoQueries;
        }
        if (keywordList.length === 0) {
          keywordList = [
            `Live-action 35mm RAW photograph of ${effectiveSubject}, wide documentary establishing view`,
            `Live-action 50mm RAW photograph of ${effectiveSubject}, crisp realistic close-up detail`,
            `Live-action 35mm RAW photograph of ${effectiveSubject}, candid medium camera angle`,
            `Live-action 35mm RAW photograph of ${effectiveSubject}, natural lighting environmental shot`,
          ];
        }
        const targetShotCount = Math.max(
          4,
          Math.min(8, Math.ceil(estimatedSeconds / effectiveClipSec))
        );
        shotPrompts = buildStoryboardPrompts({
          keywords: keywordList,
          subject: effectiveSubject,
          targetCount: targetShotCount,
          concatMode: body.videoConcatMode,
          matchToScript: false,
        });
      }

      const modelAlias = parsePollinationsModelAlias(
        body.pollinationsVideoModel
      );

      setJobProgress(
        jobId,
        16,
        `Synthesizing Gemini TTS voiceover & acquiring ${shotPrompts.length} voice-matched scenes...`
      );

      // 2. Run Gemini TTS Voiceover + Beat-Matched Visual Acquisition in parallel
      const voiceTask =
        body.voiceoverMode !== "None"
          ? synthesizeGeminiWavBuffer({
              text: finalScript,
              voice: body.voiceName || "gemini:Puck-Upbeat",
              speed: body.voiceSpeed || "1.0",
            })
          : Promise.resolve(null);

      const visualTask = (async (): Promise<{
        mode: "pollinations-video" | "pollinations-storyboard";
        perShotVideoPaths: string[];
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
          const totalBeats = shotPrompts.length;
          let fetchedVideos = 1;
          setJobProgress(
            jobId,
            24,
            `Fetched Pollinations native video clip 1/${totalBeats}...`
          );
          const perShotVideos = new Array<string>(totalBeats).fill("");
          perShotVideos[0] = firstVideoPath;

          const remaining = await mapWithConcurrency(
            shotPrompts.slice(1),
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
                20 + Math.round((fetchedVideos / totalBeats) * 34),
                `Fetched Pollinations native video clip ${fetchedVideos}/${totalBeats}...`
              );
              return r.ok ? vp : "";
            }
          );
          for (let i = 0; i < remaining.length; i++) {
            perShotVideos[i + 1] = remaining[i];
          }

          const missingIndices = perShotVideos
            .map((vp, idx) => (vp ? -1 : idx))
            .filter((idx) => idx >= 0);

          const fallbackFrames = new Array<string>(totalBeats).fill("");
          if (missingIndices.length > 0) {
            const subFrames = await fetchPollinationsStoryboardFrames({
              prompts: missingIndices.map((idx) => shotPrompts[idx]),
              photoQueries: missingIndices.map(
                (idx) => aiPhotoQueries?.[idx] || ""
              ),
              focalKeywords: missingIndices.map(
                (idx) => focalKeywords?.[idx] || ""
              ),
              subject: effectiveSubject,
              customRequirements: body.customRequirements || "",
              width,
              height,
              workDir,
            });
            missingIndices.forEach((beatIdx, subIdx) => {
              fallbackFrames[beatIdx] = subFrames[subIdx];
            });
          }

          return {
            mode: "pollinations-video",
            perShotVideoPaths: perShotVideos,
            framePaths: fallbackFrames,
            notice: `Pollinations Native Video (${modelAlias})`,
          };
        }

        // Tier 2: Multi-shot Photorealistic 35mm Live-Action Frames (1-to-1 per spoken beat)
        setJobProgress(
          jobId,
          20,
          `Acquiring ${shotPrompts.length} voice-synchronized photorealistic frames...`
        );
        const frames = await fetchPollinationsStoryboardFrames({
          prompts: shotPrompts,
          photoQueries: aiPhotoQueries,
          focalKeywords,
          subject: effectiveSubject,
          customRequirements: body.customRequirements || "",
          width,
          height,
          workDir,
          onSceneProgress: (done, total) => {
            setJobProgress(
              jobId,
              20 + Math.round((done / total) * 38),
              `Acquired voice-matched scene frame ${done} of ${total}...`
            );
          },
        });
        return {
          mode: "pollinations-storyboard",
          perShotVideoPaths: [],
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
        "Aligning clip timestamps to exact spoken voiceover pauses & syllables..."
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

      let alignedBeats: AlignedNarrativeBeat[] | undefined;
      if (narrativeBeats && narrativeBeats.length > 0) {
        alignedBeats = await alignBeatsToVoiceAudio(
          narrativeBeats,
          durationSeconds,
          hasVoiceAudio ? wavPath : undefined
        );
      }

      // 3. Build Clean Cinema ASS Subtitles locked to exact beat timings
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
          entranceAnimation: body.entranceAnimation || "Cinema Clean Dissolve",
          uppercase: body.subtitleUppercase !== false,
          fontColor: body.fontColor || "#FFFFFF",
          fontSize: body.fontSize ?? 56,
          strokeColor: body.strokeColor || "#000000",
          strokeWidth: body.strokeWidth ?? 1.5,
          shadowEnabled: body.textShadowEnabled !== false,
          shadowColor: body.textShadowColor || "#000000",
          shadowOffset: body.textShadowOffset ?? 4,
          shadowBlur: body.textShadowBlur ?? 2,
          shadowOpacity: body.textShadowOpacity ?? 80,
          bgEnabled: Boolean(body.subtitleBackgroundEnabled),
          bgColor: body.subtitleBackgroundColor || "#000000",
          alignedBeats,
        });
        fs.writeFileSync(assPath, assContent, "utf8");
      }

      // 4. Render 30fps Scene Clips with Exact Per-Beat Voice-Synchronized Durations
      const transitionMode = body.videoTransitionMode || "Shuffle";
      const photoAnimationMode =
        body.photoAnimationMode || "Auto Dynamic (Zoom + Slide + Pan)";
      const sampleTrans = resolveXfadeTransition(transitionMode, 0);
      const fitMode = body.sourceFrameFit || "Fill (center crop)";
      const concurrency = Math.max(
        1,
        Math.min(4, parseInt(body.clipRenderingConcurrency || "2", 10) || 2)
      );

      const activeSceneCount = Math.max(1, shotPrompts.length);

      const beatStarts: number[] = [];
      const beatDurations: number[] = [];
      if (alignedBeats && alignedBeats.length === activeSceneCount) {
        for (let i = 0; i < activeSceneCount; i++) {
          beatStarts.push(alignedBeats[i].startSec);
          beatDurations.push(alignedBeats[i].durationSec);
        }
      } else {
        const uniformDur = durationSeconds / activeSceneCount;
        for (let i = 0; i < activeSceneCount; i++) {
          beatStarts.push(i * uniformDur);
          beatDurations.push(uniformDur);
        }
      }

      const minBeatDur = Math.min(...beatDurations);
      const effTransDur =
        transitionMode === "None" ||
        sampleTrans.duration <= 0 ||
        activeSceneCount <= 1
          ? 0
          : Math.min(sampleTrans.duration, Math.max(0.16, minBeatDur * 0.34));

      const clipRenderDurations = beatDurations.map((bDur, idx) => {
        if (effTransDur <= 0 || activeSceneCount <= 1) {
          return Math.max(0.6, bDur);
        }
        if (idx === 0 || idx === activeSceneCount - 1) {
          return Math.max(0.8, bDur + effTransDur * 0.5);
        }
        return Math.max(0.9, bDur + effTransDur);
      });

      const isStaticPhotoMode =
        photoAnimationMode === "None (Static Photo)" ||
        photoAnimationMode === "None";

      setJobProgress(
        jobId,
        65,
        visualResult.mode === "pollinations-video"
          ? `Normalizing ${activeSceneCount} voice-synced video clips to ${VIDEO_RENDER_FPS}fps...`
          : isStaticPhotoMode
          ? `Rendering ${activeSceneCount} voice-synced scenes at ${VIDEO_RENDER_FPS}fps...`
          : `Animating ${activeSceneCount} voice-synced clips (zoom, pan & slide) at ${VIDEO_RENDER_FPS}fps...`
      );

      let doneClips = 0;
      const clipPaths: string[] = await mapWithConcurrency(
        Array.from({ length: activeSceneCount }, (_, idx) => idx),
        concurrency,
        async (idx) => {
          const outClip = path.join(workDir, `scene_clip_${idx}.mp4`);
          const thisClipDur = clipRenderDurations[idx];
          const nativeVideoForBeat = visualResult.perShotVideoPaths[idx];

          if (nativeVideoForBeat && fs.existsSync(nativeVideoForBeat)) {
            await normalizeNativeVideoClip({
              inputVideoPath: nativeVideoForBeat,
              outputClipPath: outClip,
              width,
              height,
              clipDuration: thisClipDur,
              clipSpeed: clipSpeedNum,
              fitMode,
            });
          } else {
            const imgPath = visualResult.framePaths[idx];
            await renderAnimatedSceneClipFromFrame({
              imagePath: imgPath,
              outputClipPath: outClip,
              width,
              height,
              clipDuration: thisClipDur,
              clipSpeed: clipSpeedNum,
              fitMode,
              photoAnimationMode,
              videoTransitionMode: transitionMode,
              shotIndex: idx,
            });
          }

          doneClips += 1;
          setJobProgress(
            jobId,
            65 + Math.round((doneClips / activeSceneCount) * 23),
            `Rendered voice-matched clip ${doneClips}/${activeSceneCount}...`
          );
          return outClip;
        }
      );

      // 5. Final FFmpeg Assembly + Clean ASS Subtitles + Voiceover & BGM Mix
      setJobProgress(
        jobId,
        90,
        "Encoding 30fps H.264 MP4 with synchronized clips, subtitles & audio..."
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

        for (let i = 0; i < n; i++) {
          if (effTransDur <= 0 && n > 1) {
            filterParts.push(
              `[${i}:v]trim=duration=${beatDurations[i].toFixed(
                3
              )},setpts=PTS-STARTPTS,fps=${VIDEO_RENDER_FPS},settb=AVTB,setsar=1,format=yuv420p[vnorm${i}]`
            );
          } else {
            filterParts.push(
              `[${i}:v]fps=${VIDEO_RENDER_FPS},settb=AVTB,setsar=1,format=yuv420p[vnorm${i}]`
            );
          }
        }

        let lastVideoLabel = "vnorm0";
        if (n > 1) {
          if (effTransDur <= 0) {
            const concatInputs = Array.from(
              { length: n },
              (_, idx) => `[vnorm${idx}]`
            ).join("");
            filterParts.push(`${concatInputs}concat=n=${n}:v=1:a=0[vcat]`);
            lastVideoLabel = "vcat";
          } else {
            let prevOffset = 0;
            for (let i = 1; i < n; i++) {
              const { transition } = resolveXfadeTransition(
                transitionMode,
                i - 1
              );
              const idealOffset = beatStarts[i] - effTransDur * 0.5;
              const offset = Math.max(
                i === 1 ? 0.15 : prevOffset + effTransDur + 0.08,
                idealOffset
              );
              prevOffset = offset;
              const nextLabel = `vxf${i}`;
              filterParts.push(
                `[${lastVideoLabel}][vnorm${i}]xfade=transition=${transition}:duration=${effTransDur.toFixed(
                  3
                )}:offset=${offset.toFixed(3)}[${nextLabel}]`
              );
              lastVideoLabel = nextLabel;
            }
          }
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
          )}:r=${VIDEO_RENDER_FPS}`
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
        String(VIDEO_RENDER_FPS),
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
