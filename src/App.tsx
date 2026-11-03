import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  Settings,
  Wand2,
  RotateCcw,
  Eye,
  AudioLines,
  FileText,
  Download,
  Upload,
  Loader2,
  Info,
  Play,
  CheckCircle2,
  AlertCircle,
  Film,
} from "lucide-react";
import {
  DEFAULT_DASHBOARD_SETTINGS,
  DEFAULT_SUBTITLE_SETTINGS,
  DEFAULT_SYSTEM_PROMPT,
  GEMINI_TTS_VOICES,
  PHOTO_ANIMATION_MODES,
  POLLINATIONS_VIDEO_MODELS,
  SCRIPT_LANGUAGES,
  SUBTITLE_FONTS,
  VIDEO_TRANSITION_MODES,
  type DashboardSettings,
  type LibraryVideo,
} from "./types/dashboard";
import {
  CheckboxField,
  ColorPickerField,
  Expander,
  FieldLabelRow,
  NumberStepper,
  SegmentedControl,
  SelectField,
  SliderField,
} from "./components/FormPrimitives";
import { LibraryModal } from "./components/LibraryModal";
import { PromptPreviewModal, SettingsModal } from "./components/Dialogs";

export default function App() {
  const [settings, setSettings] = useState<DashboardSettings>(
    DEFAULT_DASHBOARD_SETTINGS
  );

  // Modals
  const [libraryOpen, setLibraryOpen] = useState<boolean>(false);
  const [selectedLibraryVideoId, setSelectedLibraryVideoId] = useState<
    string | null
  >(null);
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [promptPreviewOpen, setPromptPreviewOpen] = useState<boolean>(false);
  const [autoOpenLibrary, setAutoOpenLibrary] = useState<boolean>(true);

  // Generated Video Library
  const [libraryVideos, setLibraryVideos] = useState<LibraryVideo[]>([]);

  // Generation & Preview State
  const [isGeneratingScript, setIsGeneratingScript] = useState<boolean>(false);
  const [isGeneratingKeywords, setIsGeneratingKeywords] =
    useState<boolean>(false);
  const [isGeneratingVoiceSample, setIsGeneratingVoiceSample] =
    useState<boolean>(false);
  const [isGeneratingFullVoice, setIsGeneratingFullVoice] =
    useState<boolean>(false);
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [uploadedVoiceFileName, setUploadedVoiceFileName] =
    useState<string>("");
  const [uploadedBgmFileName, setUploadedBgmFileName] = useState<string>("");

  const [pipelineState, setPipelineState] = useState<{
    running: boolean;
    stepLabel: string;
    progress: number;
  }>({ running: false, stepLabel: "", progress: 0 });

  const [feedbackBanner, setFeedbackBanner] = useState<{
    type: "info" | "success" | "error";
    message: string;
  } | null>(null);

  const presetInputRef = useRef<HTMLInputElement | null>(null);
  const scriptIsAutoRef = useRef<boolean>(true);
  const keywordsAreAutoRef = useRef<boolean>(true);
  const generatedPromptSigRef = useRef<string>("");

  const computePromptSig = (s: DashboardSettings): string =>
    JSON.stringify([
      s.videoSubject.trim(),
      s.scriptLanguage,
      s.paragraphCount,
      s.customScriptRequirements.trim(),
      s.systemPrompt.trim(),
    ]);

  // Enforce dark mode class permanently & load existing Library items
  useEffect(() => {
    document.documentElement.classList.add("dark");
    fetch("/api/library")
      .then((r) => r.json())
      .then((data: { videos?: LibraryVideo[] }) => {
        if (Array.isArray(data.videos)) {
          setLibraryVideos(data.videos);
        }
      })
      .catch(() => {});
  }, []);

  const updateField = <K extends keyof DashboardSettings>(
    key: K,
    value: DashboardSettings[K]
  ) => {
    if (key === "videoScript") {
      const isEmpty = String(value).trim() === "";
      scriptIsAutoRef.current = isEmpty;
      setSettings((prev) => ({
        ...prev,
        videoScript: String(value),
        videoKeywords: keywordsAreAutoRef.current ? "" : prev.videoKeywords,
      }));
      return;
    }

    if (key === "videoKeywords") {
      keywordsAreAutoRef.current = String(value).trim() === "";
      setSettings((prev) => ({ ...prev, videoKeywords: String(value) }));
      return;
    }

    const isPromptField =
      key === "videoSubject" ||
      key === "customScriptRequirements" ||
      key === "scriptLanguage" ||
      key === "paragraphCount" ||
      key === "systemPrompt";

    setSettings((prev) => {
      const next = { ...prev, [key]: value };
      if (isPromptField) {
        if (scriptIsAutoRef.current && prev.videoScript) {
          next.videoScript = "";
        }
        if (keywordsAreAutoRef.current && prev.videoKeywords) {
          next.videoKeywords = "";
        }
      }
      return next;
    });
  };

  // Word count & speaking duration estimate
  const scriptWordCount = useMemo(() => {
    const clean = settings.videoScript.trim();
    if (!clean) return 0;
    return clean.split(/\s+/).filter(Boolean).length;
  }, [settings.videoScript]);

  const estimatedVoiceSeconds = useMemo(() => {
    if (scriptWordCount === 0) return 0;
    const speed = parseFloat(settings.voiceSpeed) || 1.0;
    return Math.max(3, Math.round(scriptWordCount / (2.5 * speed)));
  }, [scriptWordCount, settings.voiceSpeed]);

  const subtitlePreviewLine = useMemo(() => {
    const raw =
      settings.videoScript
        .trim()
        .split(/[.!?。！？\r\n]+/)
        .map((s) => s.trim())
        .find(Boolean) ||
      settings.videoSubject.trim() ||
      "The snake strikes in silence";
    const words = raw.split(/\s+/).filter(Boolean);
    const snippet =
      settings.subtitleDisplayMode === "Word by word"
        ? words.slice(0, 3).join(" ")
        : words.slice(0, 6).join(" ");
    return settings.subtitleUppercase ? snippet.toUpperCase() : snippet;
  }, [
    settings.videoScript,
    settings.videoSubject,
    settings.subtitleDisplayMode,
    settings.subtitleUppercase,
  ]);

  // Handlers: Gemini Script & Keywords
  const handleGenerateScriptAndKeywords = async () => {
    const subject = settings.videoSubject.trim();
    if (!subject) {
      setFeedbackBanner({
        type: "error",
        message: "Please enter a Video Subject before generating with Gemini AI.",
      });
      return;
    }

    setFeedbackBanner(null);
    setIsGeneratingScript(true);
    try {
      const res = await fetch("/api/generate-script", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject,
          language: settings.scriptLanguage,
          paragraphCount: settings.paragraphCount,
          customRequirements: settings.customScriptRequirements,
          systemPrompt: settings.systemPrompt,
        }),
      });
      const data = (await res.json()) as {
        script?: string;
        keywords?: string[];
        error?: string;
      };
      if (!res.ok) {
        throw new Error(data.error || "Failed to generate script.");
      }
      if (data.script) {
        scriptIsAutoRef.current = true;
      }
      if (data.keywords && data.keywords.length > 0) {
        keywordsAreAutoRef.current = true;
      }
      generatedPromptSigRef.current = computePromptSig(settings);
      setSettings((prev) => ({
        ...prev,
        videoScript: data.script || prev.videoScript,
        videoKeywords:
          data.keywords && data.keywords.length > 0
            ? data.keywords.join(" | ")
            : prev.videoKeywords,
      }));
      setFeedbackBanner({
        type: "success",
        message: "Generated script and Pollinations AI visual keywords via Gemini.",
      });
    } catch (err) {
      setFeedbackBanner({
        type: "error",
        message:
          err instanceof Error
            ? err.message
            : "Could not reach Gemini script generation service.",
      });
    } finally {
      setIsGeneratingScript(false);
    }
  };

  const handleGenerateKeywordsOnly = async () => {
    const basis = settings.videoScript.trim() || settings.videoSubject.trim();
    if (!basis) {
      setFeedbackBanner({
        type: "error",
        message:
          "Please enter a Video Subject or Video Script before generating keywords.",
      });
      return;
    }

    setFeedbackBanner(null);
    setIsGeneratingKeywords(true);
    try {
      const res = await fetch("/api/generate-keywords", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: settings.videoSubject,
          script: settings.videoScript,
          customRequirements: settings.customScriptRequirements,
        }),
      });
      const data = (await res.json()) as {
        keywords?: string[];
        error?: string;
      };
      if (!res.ok) {
        throw new Error(data.error || "Failed to generate keywords.");
      }
      if (data.keywords && data.keywords.length > 0) {
        keywordsAreAutoRef.current = true;
        generatedPromptSigRef.current = computePromptSig(settings);
        setSettings((prev) => ({
          ...prev,
          videoKeywords: data.keywords!.join(" | "),
        }));
      }
      setFeedbackBanner({
        type: "success",
        message: "Extracted visual scene keywords via Gemini.",
      });
    } catch (err) {
      setFeedbackBanner({
        type: "error",
        message:
          err instanceof Error ? err.message : "Failed to extract keywords.",
      });
    } finally {
      setIsGeneratingKeywords(false);
    }
  };

  // Handlers: Gemini TTS Sample & Full Preview
  const synthesizeSpeech = async (text: string): Promise<string | null> => {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        voice: settings.voiceName,
        speed: settings.voiceSpeed,
      }),
    });
    const data = (await res.json()) as {
      audioBase64?: string;
      mimeType?: string;
      error?: string;
    };
    if (!res.ok || !data.audioBase64) {
      throw new Error(data.error || "Gemini TTS synthesis failed.");
    }
    return `data:${data.mimeType || "audio/wav"};base64,${data.audioBase64}`;
  };

  const handlePlayVoiceSample = async () => {
    setFeedbackBanner(null);
    setIsGeneratingVoiceSample(true);
    try {
      const sampleLine = settings.videoSubject.trim()
        ? `Here is a Gemini voice preview for your video about ${settings.videoSubject.trim()}.`
        : "Hello! This is a Gemini Text-to-Speech voice sample.";
      const dataUrl = await synthesizeSpeech(sampleLine);
      if (dataUrl) {
        setPreviewAudioUrl(dataUrl);
      }
    } catch (err) {
      setFeedbackBanner({
        type: "error",
        message:
          err instanceof Error ? err.message : "Voice sample synthesis failed.",
      });
    } finally {
      setIsGeneratingVoiceSample(false);
    }
  };

  const handleFullVoicePreview = async () => {
    const scriptText = settings.videoScript.trim();
    if (!scriptText) {
      setFeedbackBanner({
        type: "error",
        message:
          "Please enter or generate a Video Script first to preview the full voiceover.",
      });
      return;
    }
    setFeedbackBanner(null);
    setIsGeneratingFullVoice(true);
    try {
      const dataUrl = await synthesizeSpeech(scriptText);
      if (dataUrl) {
        setPreviewAudioUrl(dataUrl);
      }
    } catch (err) {
      setFeedbackBanner({
        type: "error",
        message:
          err instanceof Error ? err.message : "Full voiceover preview failed.",
      });
    } finally {
      setIsGeneratingFullVoice(false);
    }
  };

  // Preset Export & Import
  const handleExportPreset = () => {
    const blob = new Blob([JSON.stringify(settings, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "video-settings-preset.json";
    a.click();
    URL.revokeObjectURL(url);
    setFeedbackBanner({
      type: "info",
      message:
        "Exported current script, video, audio, and subtitle settings to JSON.",
    });
  };

  const handleImportPreset = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(
          String(reader.result)
        ) as Partial<DashboardSettings>;
        setSettings((prev) => ({
          ...prev,
          ...parsed,
          videoSource: "Pollinations AI",
          ttsService: "Google Gemini TTS",
        }));
        setFeedbackBanner({
          type: "success",
          message: `Imported settings preset from ${file.name}.`,
        });
      } catch {
        setFeedbackBanner({
          type: "error",
          message: "Invalid JSON settings preset file.",
        });
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  // Delete video from Library
  const handleDeleteLibraryVideo = async (id: string) => {
    try {
      const res = await fetch(`/api/library/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setLibraryVideos((prev) => prev.filter((v) => v.id !== id));
      }
    } catch {
      setLibraryVideos((prev) => prev.filter((v) => v.id !== id));
    }
  };

  // Primary Action: Generate MP4 Video via Pollinations AI + Gemini TTS + FFmpeg
  const handleGenerateVideo = async () => {
    const subject = settings.videoSubject.trim();
    const existingScript = settings.videoScript.trim();

    if (!subject && !existingScript) {
      setFeedbackBanner({
        type: "error",
        message:
          "Please enter a Video Subject or Video Script before starting video generation.",
      });
      return;
    }

    setFeedbackBanner(null);
    setPipelineState({
      running: true,
      stepLabel: "Starting Pollinations AI & Gemini video synthesis...",
      progress: 5,
    });

    try {
      const count = Math.max(
        1,
        Math.min(5, parseInt(settings.videosPerRun, 10) || 1)
      );
      let latestVideo: LibraryVideo | null = null;
      let latestNotice = "";
      const currentPromptSig = computePromptSig(settings);
      const isPromptChanged = generatedPromptSigRef.current !== currentPromptSig;

      // Never reuse stale AI-generated script/keywords when the user changed the Video Subject or requirements
      let activeScript =
        scriptIsAutoRef.current && isPromptChanged ? "" : settings.videoScript;
      let activeKeywords =
        keywordsAreAutoRef.current && isPromptChanged
          ? ""
          : settings.videoKeywords;

      for (let runIdx = 0; runIdx < count; runIdx++) {
        const prefix = count > 1 ? `[Video ${runIdx + 1}/${count}] ` : "";

        const startRes = await fetch("/api/render-video", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            subject: settings.videoSubject,
            script: activeScript,
            keywords: activeKeywords,
            language: settings.scriptLanguage,
            paragraphCount: settings.paragraphCount,
            customRequirements: settings.customScriptRequirements,
            systemPrompt: settings.systemPrompt,
            pollinationsVideoModel: settings.pollinationsVideoModel,
            videoConcatMode: settings.videoConcatMode,
            matchMaterialsToScript: settings.matchMaterialsToScript,
            photoAnimationMode: settings.photoAnimationMode,
            videoTransitionMode: settings.videoTransitionMode,
            aspectRatio: settings.videoAspectRatio,
            sourceFrameFit: settings.sourceFrameFit,
            maxClipDuration: settings.maxClipDuration,
            clipSpeed: settings.clipSpeed,
            clipRenderingConcurrency: settings.clipRenderingConcurrency,
            voiceoverMode: settings.voiceoverMode,
            voiceName: settings.voiceName,
            voiceVolume: settings.voiceVolume,
            voiceSpeed: settings.voiceSpeed,
            bgmSource: settings.bgmSource,
            bgmVolume: settings.bgmVolume,
            subtitleEnabled: settings.subtitleEnabled,
            subtitleUppercase: settings.subtitleUppercase,
            subtitlePosition: settings.subtitlePosition,
            customPositionPercent: settings.customPositionPercent,
            subtitleDisplayMode: settings.subtitleDisplayMode,
            entranceAnimation: settings.entranceAnimation,
            fontColor: settings.fontColor,
            fontSize: settings.fontSize,
            strokeColor: settings.strokeColor,
            strokeWidth: settings.strokeWidth,
            textShadowEnabled: settings.textShadowEnabled,
            textShadowColor: settings.textShadowColor,
            textShadowOffset: settings.textShadowOffset,
            textShadowBlur: settings.textShadowBlur,
            textShadowOpacity: settings.textShadowOpacity,
            subtitleBackgroundEnabled: settings.subtitleBackgroundEnabled,
            subtitleBackgroundColor: settings.subtitleBackgroundColor,
          }),
        });

        const startRaw = await startRes.text();
        let startData: {
          jobId?: string;
          progress?: number;
          stepLabel?: string;
          error?: string;
        } = {};
        try {
          startData = JSON.parse(startRaw);
        } catch {
          throw new Error("Unexpected response while starting video render.");
        }

        if (!startRes.ok || !startData.jobId) {
          throw new Error(startData.error || "Failed to start video rendering.");
        }

        const jobId = startData.jobId;
        const initialOverall = Math.max(
          1,
          Math.round(((runIdx * 100 + (startData.progress ?? 5)) / count))
        );
        setPipelineState({
          running: true,
          stepLabel: `${prefix}${startData.stepLabel || "Starting video generation..."}`,
          progress: initialOverall,
        });

        // Poll server-side render job until 100% completed
        let completedData: {
          video?: LibraryVideo;
          script?: string;
          keywords?: string[];
          notice?: string;
        } | null = null;
        let transientPollErrors = 0;

        while (!completedData) {
          await new Promise((resolve) => window.setTimeout(resolve, 550));

          let pollRes: Response;
          try {
            pollRes = await fetch(
              `/api/render-status/${encodeURIComponent(jobId)}`
            );
          } catch {
            transientPollErrors += 1;
            if (transientPollErrors > 10) {
              throw new Error(
                "Lost connection while checking video generation status."
              );
            }
            continue;
          }

          if (pollRes.status === 404) {
            throw new Error("Render job expired or server restarted. Please click Generate Video again.");
          }

          const rawPoll = await pollRes.text();
          let pollData: {
            status?: "running" | "completed" | "failed";
            progress?: number;
            stepLabel?: string;
            video?: LibraryVideo;
            script?: string;
            keywords?: string[];
            notice?: string;
            error?: string;
          } = {};

          try {
            pollData = JSON.parse(rawPoll);
            transientPollErrors = 0;
          } catch {
            transientPollErrors += 1;
            if (transientPollErrors > 10) {
              throw new Error("Invalid status response from video renderer.");
            }
            continue;
          }

          if (pollData.status === "failed") {
            throw new Error(pollData.error || "Video rendering failed.");
          }

          const rawPct = Math.max(5, Math.min(100, pollData.progress ?? 10));
          const overallPct = Math.round(((runIdx * 100 + rawPct) / count));
          setPipelineState({
            running: true,
            stepLabel: `${prefix}${pollData.stepLabel || "Rendering MP4 video..."}`,
            progress: overallPct,
          });

          if (pollData.status === "completed" && pollData.video) {
            completedData = pollData;
          }
        }

        if (!activeScript.trim() && completedData.script) {
          activeScript = completedData.script;
          scriptIsAutoRef.current = true;
        }
        if (
          completedData.keywords &&
          completedData.keywords.length > 0 &&
          (!activeKeywords.trim() ||
            keywordsAreAutoRef.current ||
            settings.matchMaterialsToScript)
        ) {
          activeKeywords = completedData.keywords.join(" | ");
          keywordsAreAutoRef.current = true;
        }
        generatedPromptSigRef.current = currentPromptSig;
        setSettings((prev) => ({
          ...prev,
          videoScript: activeScript || prev.videoScript,
          videoKeywords: activeKeywords || prev.videoKeywords,
        }));

        latestVideo = completedData.video!;
        if (completedData.notice) {
          latestNotice = completedData.notice;
        }
        setLibraryVideos((prev) => [
          completedData!.video!,
          ...prev.filter((x) => x.id !== completedData!.video!.id),
        ]);
      }

      setPipelineState({
        running: true,
        stepLabel: "Completed! Opening Library...",
        progress: 100,
      });
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      setPipelineState({ running: false, stepLabel: "", progress: 100 });

      if (latestVideo) {
        setSelectedLibraryVideoId(latestVideo.id);
        if (autoOpenLibrary) {
          setLibraryOpen(true);
        }
        setFeedbackBanner({
          type: "success",
          message: latestNotice
            ? `Generated "${latestVideo.subject}" (${latestNotice}). Ready in Library.`
            : `Generated "${latestVideo.subject}" with Pollinations AI & Gemini. Ready to watch and download in Library.`,
        });
      }
    } catch (err) {
      setPipelineState({ running: false, stepLabel: "", progress: 0 });
      setFeedbackBanner({
        type: "error",
        message:
          err instanceof Error
            ? err.message
            : "Video synthesis pipeline failed.",
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#0e1117] text-[#fafafa]">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 pt-3 pb-10">
        {/* Top Bar: Right-aligned Library + Settings */}
        <header className="flex items-center justify-end gap-2.5 py-2 mb-3">
          <div className="flex items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => setLibraryOpen(true)}
              className="min-h-[38px] px-3.5 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[14px] font-normal text-[#fafafa] inline-flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Film className="w-4 h-4 shrink-0" />
              <span>
                Library
                {libraryVideos.length > 0 ? ` (${libraryVideos.length})` : ""}
              </span>
            </button>

            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="min-h-[38px] px-3.5 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[14px] font-normal text-[#fafafa] inline-flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Settings className="w-4 h-4 shrink-0" />
              <span>Settings</span>
            </button>
          </div>
        </header>

        {/* Status / Feedback Callout */}
        {feedbackBanner && (
          <div
            role="status"
            className={`mb-4 px-4 py-3 rounded-lg border text-[13.5px] flex items-center justify-between gap-3 ${
              feedbackBanner.type === "error"
                ? "bg-[#dc2626]/15 border-[#dc2626]/30 text-[#fca5a5]"
                : feedbackBanner.type === "success"
                ? "bg-[#16a34a]/15 border-[#16a34a]/30 text-[#86efac]"
                : "bg-[#0284c7]/15 border-[#0284c7]/30 text-[#7dd3fc]"
            }`}
          >
            <div className="flex items-center gap-2 min-w-0">
              {feedbackBanner.type === "error" ? (
                <AlertCircle className="w-4 h-4 shrink-0" />
              ) : feedbackBanner.type === "success" ? (
                <CheckCircle2 className="w-4 h-4 shrink-0" />
              ) : (
                <Info className="w-4 h-4 shrink-0" />
              )}
              <span>{feedbackBanner.message}</span>
            </div>
            <button
              type="button"
              onClick={() => setFeedbackBanner(null)}
              className="text-[12px] underline underline-offset-2 opacity-75 hover:opacity-100 whitespace-nowrap shrink-0 cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Main 4-Column Settings Grid */}
        <main className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 items-start">
          {/* COLUMN 1: Video Script Settings (Gemini API) */}
          <section className="rounded-lg border border-white/14 bg-[#0e1117] p-4 space-y-4">
            <h2 className="text-[15.5px] font-bold text-[#fafafa] tracking-[0.01em]">
              Video Script Settings
            </h2>

            <div>
              <FieldLabelRow
                htmlFor="video-subject-input"
                label="Video Subject"
                tooltip="Gemini AI will generate the video script and Pollinations AI scene keywords from this topic."
              />
              <textarea
                id="video-subject-input"
                rows={3}
                value={settings.videoSubject}
                onChange={(e) => updateField("videoSubject", e.target.value)}
                placeholder="Example: How AI is changing everyday life"
                className="w-full rounded-lg bg-[#262730] border border-white/15 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none p-3 text-[14px] leading-relaxed text-[#fafafa] placeholder:text-[#fafafa]/40 resize-y"
              />
            </div>

            <SelectField
              label="Script Language"
              value={settings.scriptLanguage}
              options={SCRIPT_LANGUAGES}
              onChange={(val) => updateField("scriptLanguage", val)}
            />

            <Expander title="Advanced Script Settings" variant="subtle">
              <NumberStepper
                label="Script Paragraph Count"
                value={settings.paragraphCount}
                min={1}
                max={10}
                onChange={(val) => updateField("paragraphCount", val)}
              />

              <div>
                <FieldLabelRow
                  htmlFor="custom-script-req"
                  label="Custom Script Requirements"
                />
                <textarea
                  id="custom-script-req"
                  rows={3}
                  value={settings.customScriptRequirements}
                  onChange={(e) =>
                    updateField("customScriptRequirements", e.target.value)
                  }
                  placeholder="Example: use a lighter tone, fit TikTok style, target young viewers, make the opening more suspenseful"
                  className="w-full rounded-lg bg-[#161922] border border-white/15 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none p-2.5 text-[13.5px] leading-relaxed text-[#fafafa] placeholder:text-[#fafafa]/40 resize-y"
                />
              </div>

              <div>
                <FieldLabelRow
                  htmlFor="system-prompt-textarea"
                  label="System Prompt"
                />
                <textarea
                  id="system-prompt-textarea"
                  rows={6}
                  value={settings.systemPrompt}
                  onChange={(e) => updateField("systemPrompt", e.target.value)}
                  className="w-full rounded-lg bg-[#161922] border border-white/15 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none p-2.5 font-mono text-[12px] leading-relaxed text-[#fafafa] resize-y"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-0.5">
                <button
                  type="button"
                  onClick={() =>
                    updateField("systemPrompt", DEFAULT_SYSTEM_PROMPT)
                  }
                  className="min-h-[38px] px-2.5 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[12.5px] text-[#fafafa] inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap truncate cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">Restore Default Prompt</span>
                </button>
                <button
                  type="button"
                  onClick={() => setPromptPreviewOpen(true)}
                  className="min-h-[38px] px-2.5 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[12.5px] text-[#fafafa] inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap truncate cursor-pointer"
                >
                  <Eye className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">Preview Final Prompt</span>
                </button>
              </div>
            </Expander>

            <button
              type="button"
              disabled={isGeneratingScript}
              onClick={handleGenerateScriptAndKeywords}
              className="w-full min-h-[40px] px-3 py-2 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] disabled:opacity-55 text-[14px] font-normal text-[#fafafa] inline-flex items-center justify-center gap-2 transition-colors whitespace-nowrap truncate cursor-pointer"
            >
              {isGeneratingScript ? (
                <Loader2 className="w-4 h-4 shrink-0 animate-spin text-[#ff4b4b]" />
              ) : (
                <Wand2 className="w-4 h-4 shrink-0" />
              )}
              <span className="truncate">
                {isGeneratingScript
                  ? "Generating Script & Keywords..."
                  : "Generate Script & Keywords with AI"}
              </span>
            </button>

            <div>
              <div className="flex items-center justify-between gap-2">
                <FieldLabelRow
                  htmlFor="video-script-textarea"
                  label="Video Script (Optional)"
                  tooltip="Leave blank to generate automatically with Gemini from the video subject. Changing Video Subject automatically refreshes AI-generated scripts."
                />
                {settings.videoScript && (
                  <button
                    type="button"
                    onClick={() => updateField("videoScript", "")}
                    className="mb-1.5 text-[12px] text-[#fafafa]/55 hover:text-[#ff4b4b] transition-colors cursor-pointer"
                  >
                    Clear
                  </button>
                )}
              </div>
              <textarea
                id="video-script-textarea"
                rows={5}
                value={settings.videoScript}
                onChange={(e) => updateField("videoScript", e.target.value)}
                placeholder="Leave blank to auto-generate a fresh script from your Video Subject..."
                className="w-full rounded-lg bg-[#262730] border border-white/15 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none p-3 text-[14px] leading-relaxed text-[#fafafa] placeholder:text-[#fafafa]/35 resize-y"
              />
            </div>

            <button
              type="button"
              disabled={isGeneratingKeywords}
              onClick={handleGenerateKeywordsOnly}
              className="w-full min-h-[40px] px-3 py-2 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] disabled:opacity-55 text-[14px] font-normal text-[#fafafa] inline-flex items-center justify-center gap-2 transition-colors whitespace-nowrap truncate cursor-pointer"
            >
              {isGeneratingKeywords ? (
                <Loader2 className="w-4 h-4 shrink-0 animate-spin text-[#ff4b4b]" />
              ) : (
                <Wand2 className="w-4 h-4 shrink-0" />
              )}
              <span className="truncate">
                {isGeneratingKeywords
                  ? "Generating Keywords..."
                  : "Generate Keywords with AI"}
              </span>
            </button>

            <div>
              <div className="flex items-center justify-between gap-2">
                <FieldLabelRow
                  htmlFor="video-keywords-input"
                  label="Video Keywords (English, Optional)"
                  tooltip="Leave blank to extract scene-by-scene visual prompts automatically with Gemini."
                />
                {settings.videoKeywords && (
                  <button
                    type="button"
                    onClick={() => updateField("videoKeywords", "")}
                    className="mb-1.5 text-[12px] text-[#fafafa]/55 hover:text-[#ff4b4b] transition-colors cursor-pointer"
                  >
                    Clear
                  </button>
                )}
              </div>
              <input
                id="video-keywords-input"
                type="text"
                value={settings.videoKeywords}
                onChange={(e) => updateField("videoKeywords", e.target.value)}
                placeholder="Leave blank to auto-extract Pollinations AI visual prompts..."
                className="w-full min-h-[40px] px-3 py-2 rounded-lg bg-[#262730] border border-white/15 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none text-[14px] text-[#fafafa] placeholder:text-[#fafafa]/35"
              />
            </div>
          </section>

          {/* COLUMN 2: Video Settings (Pollinations AI Exclusively) */}
          <section className="rounded-lg border border-white/14 bg-[#0e1117] p-4 space-y-4">
            <h2 className="text-[15.5px] font-bold text-[#fafafa] tracking-[0.01em]">
              Video Settings
            </h2>

            <div className="space-y-1.5">
              <SelectField
                label="Video Source"
                value={settings.videoSource}
                options={["Pollinations AI"]}
                onChange={() => updateField("videoSource", "Pollinations AI")}
              />
            </div>

            <div className="space-y-1.5">
              <SelectField
                label="Pollinations Video Model"
                value={settings.pollinationsVideoModel}
                options={POLLINATIONS_VIDEO_MODELS}
                tooltip="Calls gen.pollinations.ai/video/{prompt} when Paid Pollen is active, or acquires high-resolution photorealistic live-action storyboard frames."
                onChange={(val) => updateField("pollinationsVideoModel", val)}
              />
              <p className="text-[12.5px] leading-relaxed text-[#fafafa]/60">
                Animates photorealistic scenes with smooth 30fps camera motion (zoom-in, zoom-out, pan &amp; slide) and native Pollinations AI clips into H.264 MP4 videos.
              </p>
            </div>

            <SelectField
              label="Video Concatenation Mode"
              value={settings.videoConcatMode}
              options={[
                "Sequential Concatenation",
                "Random Concatenation (Recommended)",
              ]}
              onChange={(val) => updateField("videoConcatMode", val)}
            />

            <CheckboxField
              label="Match Visuals to Script Order"
              checked={settings.matchMaterialsToScript}
              onChange={(checked) =>
                updateField("matchMaterialsToScript", checked)
              }
              tooltip="Synchronizes each clip's visual subject and transition timing directly to the spoken sentences, entities, and voiceover pauses."
            />

            <SelectField
              label="Photo Clip Animation"
              value={settings.photoAnimationMode}
              options={PHOTO_ANIMATION_MODES}
              tooltip="Applies continuous subpixel camera motion (zoom-in, zoom-out, horizontal slide, vertical pan, or Ken Burns drift) to every generated photo clip."
              onChange={(val) => updateField("photoAnimationMode", val)}
            />

            <SelectField
              label="Video Transition Mode"
              value={settings.videoTransitionMode}
              options={VIDEO_TRANSITION_MODES}
              tooltip="Selects the visual transition applied between video scenes."
              onChange={(val) => updateField("videoTransitionMode", val)}
            />

            <SelectField
              label="Video Aspect Ratio"
              value={settings.videoAspectRatio}
              options={["Portrait 9:16", "Landscape 16:9", "Square 1:1"]}
              onChange={(val) => updateField("videoAspectRatio", val)}
            />

            <SelectField
              label="Source Frame Fit"
              value={settings.sourceFrameFit}
              options={["Fill (center crop)", "Fit (black bars)"]}
              tooltip="Fill uses the entire output canvas and crops excess edges. Fit preserves the complete source frame and adds black bars when needed."
              onChange={(val) => updateField("sourceFrameFit", val)}
            />

            <SelectField
              label="Maximum Clip Duration (seconds)"
              value={settings.maxClipDuration}
              options={[
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
              ]}
              tooltip="Controls the target duration of each Pollinations AI scene segment."
              onChange={(val) => updateField("maxClipDuration", val)}
            />

            <SliderField
              label="Clip Speed"
              value={settings.clipSpeed}
              min={0.5}
              max={2.0}
              step={0.05}
              formatValue={(v) => `${v.toFixed(2)}x`}
              tooltip="Adjusts visual material playback speed while keeping voiceover and subtitles synchronized."
              onChange={(val) => updateField("clipSpeed", val)}
            />

            <SelectField
              label="Videos per Run"
              value={settings.videosPerRun}
              options={["1", "2", "3", "4", "5"]}
              onChange={(val) => updateField("videosPerRun", val)}
            />

            <SelectField
              label="Video Encoder"
              value={settings.videoEncoder}
              options={[
                "Default (Recommended)",
                "libx264 (CPU)",
                "NVIDIA NVENC (h264_nvenc)",
                "AMD AMF (h264_amf)",
                "Intel QSV (h264_qsv)",
                "Windows MediaFoundation (h264_mf)",
                "macOS VideoToolbox (h264_videotoolbox)",
              ]}
              tooltip="Default uses libx264 H.264 MP4 encoding for universal browser and mobile compatibility."
              onChange={(val) => updateField("videoEncoder", val)}
            />

            <SelectField
              label="Clip Rendering Concurrency"
              value={settings.clipRenderingConcurrency}
              options={["1", "2", "4", "6", "8"]}
              tooltip="How many Pollinations AI scenes are processed concurrently."
              onChange={(val) => updateField("clipRenderingConcurrency", val)}
            />
          </section>

          {/* COLUMN 3: Audio Settings (Google Gemini TTS Exclusively) */}
          <section className="rounded-lg border border-white/14 bg-[#0e1117] p-4 space-y-4">
            <h2 className="text-[15.5px] font-bold text-[#fafafa] tracking-[0.01em]">
              Audio Settings
            </h2>

            <SegmentedControl
              label="Voiceover Mode"
              value={settings.voiceoverMode}
              options={["Auto", "Upload", "None"] as const}
              onChange={(mode) => updateField("voiceoverMode", mode)}
            />

            {settings.voiceoverMode === "Auto" && (
              <>
                <div className="space-y-1.5">
                  <SelectField
                    label="Voiceover Service"
                    value={settings.ttsService}
                    options={["Google Gemini TTS"]}
                    onChange={() =>
                      updateField("ttsService", "Google Gemini TTS")
                    }
                  />
                  <p className="text-[12.5px] leading-relaxed text-[#fafafa]/60">
                    Powered by your unified Gemini API key. Pace and expression follow natural-language voice instructions.
                  </p>
                </div>

                <SelectField
                  label="Voice (match script language)"
                  value={settings.voiceName}
                  options={GEMINI_TTS_VOICES}
                  onChange={(val) => updateField("voiceName", val)}
                />

                <SelectField
                  label="Voiceover Volume"
                  value={settings.voiceVolume}
                  options={[
                    "60%",
                    "80%",
                    "100%",
                    "120%",
                    "150%",
                    "200%",
                    "300%",
                    "400%",
                    "500%",
                  ]}
                  tooltip="Adjusts voiceover volume in the rendered MP4. 100% keeps the original level."
                  onChange={(val) => updateField("voiceVolume", val)}
                />

                <SelectField
                  label="Voiceover Speed"
                  value={settings.voiceSpeed}
                  options={[
                    "0.8",
                    "0.9",
                    "1.0",
                    "1.1",
                    "1.2",
                    "1.3",
                    "1.5",
                    "1.8",
                    "2.0",
                  ]}
                  tooltip="Applies to automatic Gemini TTS voiceover. 1.0 is normal speed."
                  onChange={(val) => updateField("voiceSpeed", val)}
                />

                <p className="text-[12.5px] leading-relaxed text-[#fafafa]/60 font-mono tabular-nums">
                  {scriptWordCount > 0
                    ? `Estimated voiceover: ~${estimatedVoiceSeconds}s (${scriptWordCount} words at ${settings.voiceSpeed}x)`
                    : "Add a video script to estimate and preview the full voiceover."}
                </p>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    disabled={isGeneratingVoiceSample}
                    onClick={handlePlayVoiceSample}
                    className="min-h-[38px] px-3 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] disabled:opacity-55 text-[13.5px] text-[#fafafa] inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap truncate cursor-pointer"
                  >
                    {isGeneratingVoiceSample ? (
                      <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin text-[#ff4b4b]" />
                    ) : (
                      <AudioLines className="w-3.5 h-3.5 shrink-0" />
                    )}
                    <span className="truncate">Voice Sample</span>
                  </button>

                  <button
                    type="button"
                    disabled={isGeneratingFullVoice}
                    onClick={handleFullVoicePreview}
                    title="Synthesizes the full script using Google Gemini TTS."
                    className="min-h-[38px] px-3 py-1.5 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] disabled:opacity-55 text-[13.5px] text-[#fafafa] inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap truncate cursor-pointer"
                  >
                    {isGeneratingFullVoice ? (
                      <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin text-[#ff4b4b]" />
                    ) : (
                      <FileText className="w-3.5 h-3.5 shrink-0" />
                    )}
                    <span className="truncate">Full Preview</span>
                  </button>
                </div>

                {previewAudioUrl && (
                  <div className="pt-1">
                    <audio
                      controls
                      autoPlay
                      src={previewAudioUrl}
                      className="w-full h-9"
                    />
                  </div>
                )}
              </>
            )}

            {settings.voiceoverMode === "Upload" && (
              <div className="rounded-lg border border-dashed border-white/25 p-3.5 bg-[#262730]/60">
                <label className="flex flex-col items-center justify-center gap-1 cursor-pointer text-center">
                  <Upload className="w-4 h-4 text-[#fafafa]/60" />
                  <span className="text-[13px] font-medium text-[#fafafa]">
                    Upload voiceover audio (.mp3, .wav, .m4a)
                  </span>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) =>
                      setUploadedVoiceFileName(e.target.files?.[0]?.name || "")
                    }
                    className="hidden"
                  />
                </label>
                {uploadedVoiceFileName && (
                  <p className="mt-2 text-[12px] text-[#fafafa]/75 truncate text-center">
                    Selected: {uploadedVoiceFileName}
                  </p>
                )}
              </div>
            )}

            <hr className="border-white/12 my-1" />

            <SelectField
              label="Background Music Source"
              value={settings.bgmSource}
              options={[
                "No Background Music",
                "Random Background Music",
                "Preset Song",
                "Custom Background Music",
              ]}
              onChange={(val) => updateField("bgmSource", val)}
            />

            {settings.bgmSource === "Custom Background Music" && (
              <div className="rounded-lg border border-dashed border-white/25 p-3 bg-[#262730]/60">
                <label className="flex flex-col items-center justify-center gap-1 cursor-pointer text-center">
                  <Upload className="w-4 h-4 text-[#fafafa]/60" />
                  <span className="text-[13px] font-medium text-[#fafafa]">
                    Upload custom BGM file (.mp3, .wav)
                  </span>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) =>
                      setUploadedBgmFileName(e.target.files?.[0]?.name || "")
                    }
                    className="hidden"
                  />
                </label>
                {uploadedBgmFileName && (
                  <p className="mt-1.5 text-[12px] text-[#fafafa]/75 truncate text-center">
                    Selected: {uploadedBgmFileName}
                  </p>
                )}
              </div>
            )}

            <SelectField
              label="Background Music Volume"
              value={settings.bgmVolume}
              disabled={settings.bgmSource === "No Background Music"}
              options={[
                "0%",
                "10%",
                "20%",
                "30%",
                "40%",
                "50%",
                "60%",
                "70%",
                "80%",
                "90%",
                "100%",
              ]}
              onChange={(val) => updateField("bgmVolume", val)}
            />
          </section>

          {/* COLUMN 4: Subtitle Settings */}
          <section className="rounded-lg border border-white/14 bg-[#0e1117] p-4 space-y-4">
            <h2 className="text-[15.5px] font-bold text-[#fafafa] tracking-[0.01em]">
              Subtitle Settings
            </h2>

            <CheckboxField
              label="Enable Subtitles"
              checked={settings.subtitleEnabled}
              onChange={(checked) => updateField("subtitleEnabled", checked)}
            />

            <CheckboxField
              label="Capital Letters (ALL CAPS)"
              checked={settings.subtitleUppercase}
              disabled={!settings.subtitleEnabled}
              tooltip="Converts all rendered video subtitles into bold capital letters."
              onChange={(checked) => updateField("subtitleUppercase", checked)}
            />

            <SelectField
              label="Font"
              value={settings.fontName}
              disabled={!settings.subtitleEnabled}
              options={SUBTITLE_FONTS}
              onChange={(val) => updateField("fontName", val)}
            />

            <SelectField
              label="Position"
              value={settings.subtitlePosition}
              disabled={!settings.subtitleEnabled}
              options={[
                "Top",
                "Center",
                "Bottom (Recommended)",
                "Upper third",
                "Custom",
              ]}
              onChange={(val) => updateField("subtitlePosition", val)}
            />

            {settings.subtitlePosition === "Custom" && (
              <SliderField
                label="Custom Vertical Position"
                value={settings.customPositionPercent}
                min={5}
                max={95}
                step={1}
                disabled={!settings.subtitleEnabled}
                formatValue={(v) => `${Math.round(v)}%`}
                onChange={(val) => updateField("customPositionPercent", val)}
              />
            )}

            <SelectField
              label="Subtitle display"
              value={settings.subtitleDisplayMode}
              disabled={!settings.subtitleEnabled}
              options={["Whole sentence", "Word by word"]}
              tooltip="Choose between full-sentence subtitles or rapid 3-word phrase pacing."
              onChange={(val) => updateField("subtitleDisplayMode", val)}
            />

            <SelectField
              label="Entrance animation"
              value={settings.entranceAnimation}
              disabled={!settings.subtitleEnabled}
              options={[
                "Cinema Clean Dissolve",
                "Subtle Ease",
                "None",
                "Kinetic Motion Ease",
                "Spring pop",
              ]}
              onChange={(val) => updateField("entranceAnimation", val)}
            />

            <div className="grid grid-cols-[auto_1fr] items-start gap-3">
              <ColorPickerField
                label="Color"
                value={settings.fontColor}
                disabled={!settings.subtitleEnabled}
                onChange={(hex) => updateField("fontColor", hex)}
              />
              <SliderField
                label="Font Size"
                value={settings.fontSize}
                min={30}
                max={100}
                step={1}
                disabled={!settings.subtitleEnabled}
                formatValue={(v) => `${Math.round(v)}`}
                onChange={(val) => updateField("fontSize", val)}
              />
            </div>

            <div className="grid grid-cols-[auto_1fr] items-start gap-3">
              <ColorPickerField
                label="Outline"
                value={settings.strokeColor}
                disabled={!settings.subtitleEnabled}
                onChange={(hex) => updateField("strokeColor", hex)}
              />
              <SliderField
                label="Outline Width"
                value={settings.strokeWidth}
                min={0}
                max={10}
                step={0.05}
                disabled={!settings.subtitleEnabled}
                formatValue={(v) => v.toFixed(2)}
                onChange={(val) => updateField("strokeWidth", val)}
              />
            </div>

            <div className="space-y-3 pt-1 border-t border-white/10">
              <CheckboxField
                label="Text Shadow"
                checked={settings.textShadowEnabled}
                disabled={!settings.subtitleEnabled}
                tooltip="Adds a customizable cinema drop shadow behind subtitle letters for high contrast over bright scenes."
                onChange={(checked) =>
                  updateField("textShadowEnabled", checked)
                }
              />

              {settings.textShadowEnabled && (
                <div className="space-y-3 pl-0.5">
                  <div className="grid grid-cols-[auto_1fr] items-start gap-3">
                    <ColorPickerField
                      label="Shadow"
                      value={settings.textShadowColor}
                      disabled={!settings.subtitleEnabled}
                      onChange={(hex) => updateField("textShadowColor", hex)}
                    />
                    <SliderField
                      label="Shadow Distance"
                      value={settings.textShadowOffset}
                      min={0.5}
                      max={12}
                      step={0.5}
                      disabled={!settings.subtitleEnabled}
                      formatValue={(v) => `${v.toFixed(1)}px`}
                      onChange={(val) => updateField("textShadowOffset", val)}
                    />
                  </div>

                  <SliderField
                    label="Shadow Softness"
                    value={settings.textShadowBlur}
                    min={0}
                    max={8}
                    step={0.5}
                    disabled={!settings.subtitleEnabled}
                    formatValue={(v) => `${v.toFixed(1)}px`}
                    onChange={(val) => updateField("textShadowBlur", val)}
                  />

                  <SliderField
                    label="Shadow Opacity"
                    value={settings.textShadowOpacity}
                    min={10}
                    max={100}
                    step={5}
                    disabled={!settings.subtitleEnabled}
                    formatValue={(v) => `${Math.round(v)}%`}
                    onChange={(val) => updateField("textShadowOpacity", val)}
                  />
                </div>
              )}
            </div>

            <div className="space-y-3 pt-1 border-t border-white/10">
              <div className="flex items-end justify-between gap-3">
                <CheckboxField
                  label="Background"
                  checked={settings.subtitleBackgroundEnabled}
                  disabled={!settings.subtitleEnabled}
                  onChange={(checked) =>
                    updateField("subtitleBackgroundEnabled", checked)
                  }
                />
                <ColorPickerField
                  label="Color"
                  value={settings.subtitleBackgroundColor}
                  disabled={
                    !settings.subtitleEnabled ||
                    !settings.subtitleBackgroundEnabled
                  }
                  onChange={(hex) =>
                    updateField("subtitleBackgroundColor", hex)
                  }
                />
              </div>

              <CheckboxField
                label="Rounded Background"
                checked={settings.roundedSubtitleBackground}
                disabled={
                  !settings.subtitleEnabled ||
                  !settings.subtitleBackgroundEnabled
                }
                tooltip="Disabled by default. When enabled, subtitle backgrounds use a rounded translucent style."
                onChange={(checked) =>
                  updateField("roundedSubtitleBackground", checked)
                }
              />
            </div>

            {settings.subtitleEnabled && (
              <div className="rounded-lg border border-white/12 bg-gradient-to-b from-[#1d2230] to-[#12151e] p-2.5">
                <div className="flex items-center justify-between text-[11.5px] text-[#fafafa]/55 mb-1.5">
                  <span>Subtitle Style Preview</span>
                  <span className="font-mono tabular-nums">
                    {settings.subtitleUppercase ? "ALL CAPS" : "Mixed Case"}
                    {settings.textShadowEnabled ? " · Shadow On" : ""}
                  </span>
                </div>
                <svg
                  viewBox="0 0 360 72"
                  className="w-full h-[64px] rounded bg-[#0b0e14]/70 overflow-hidden"
                  role="img"
                  aria-label="Live subtitle preview"
                >
                  <defs>
                    <filter
                      id="subtitle-drop-shadow"
                      x="-25%"
                      y="-25%"
                      width="150%"
                      height="150%"
                    >
                      <feDropShadow
                        dx={
                          settings.textShadowEnabled
                            ? settings.textShadowOffset
                            : 0
                        }
                        dy={
                          settings.textShadowEnabled
                            ? settings.textShadowOffset
                            : 0
                        }
                        stdDeviation={
                          settings.textShadowEnabled
                            ? settings.textShadowBlur * 0.75
                            : 0
                        }
                        floodColor={settings.textShadowColor}
                        floodOpacity={
                          settings.textShadowEnabled
                            ? settings.textShadowOpacity / 100
                            : 0
                        }
                      />
                    </filter>
                  </defs>
                  {settings.subtitleBackgroundEnabled && (
                    <rect
                      x="24"
                      y="16"
                      width="312"
                      height="40"
                      rx={settings.roundedSubtitleBackground ? 10 : 2}
                      fill={settings.subtitleBackgroundColor}
                      fillOpacity={0.75}
                      filter={
                        settings.textShadowEnabled
                          ? "url(#subtitle-drop-shadow)"
                          : undefined
                      }
                    />
                  )}
                  <text
                    x="180"
                    y="38"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill={settings.fontColor}
                    stroke={
                      settings.subtitleBackgroundEnabled ||
                      settings.strokeWidth <= 0
                        ? "none"
                        : settings.strokeColor
                    }
                    strokeWidth={
                      settings.subtitleBackgroundEnabled
                        ? 0
                        : settings.strokeWidth * 1.6
                    }
                    strokeLinejoin="round"
                    paintOrder="stroke fill"
                    fontSize={Math.max(
                      13,
                      Math.min(26, Math.round(settings.fontSize * 0.31))
                    )}
                    fontWeight="700"
                    fontFamily="sans-serif"
                    filter={
                      !settings.subtitleBackgroundEnabled &&
                      settings.textShadowEnabled
                        ? "url(#subtitle-drop-shadow)"
                        : undefined
                    }
                  >
                    {subtitlePreviewLine}
                  </text>
                </svg>
              </div>
            )}

            <button
              type="button"
              onClick={() =>
                setSettings((prev) => ({
                  ...prev,
                  ...DEFAULT_SUBTITLE_SETTINGS,
                }))
              }
              className="w-full min-h-[38px] px-3 py-2 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[13.5px] text-[#fafafa] inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap truncate cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">Restore Subtitle Defaults</span>
            </button>
          </section>
        </main>

        {/* Bottom Section: Settings Preset + Generate Video */}
        <div className="mt-4 space-y-4">
          <Expander title="Settings Preset" variant="bordered">
            <p className="text-[13.5px] leading-relaxed text-[#fafafa]/70 mb-3">
              Export the current script, video, audio, and subtitle settings to a JSON file, or import a file to restore them.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={handleExportPreset}
                className="min-h-[42px] px-4 py-2 rounded-lg border border-white/20 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[14px] text-[#fafafa] inline-flex items-center justify-center gap-2 transition-colors whitespace-nowrap cursor-pointer"
              >
                <Download className="w-4 h-4 shrink-0" />
                <span>Export Settings</span>
              </button>

              <div>
                <input
                  ref={presetInputRef}
                  type="file"
                  accept=".json,application/json"
                  onChange={handleImportPreset}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => presetInputRef.current?.click()}
                  className="w-full min-h-[42px] px-4 py-2 rounded-lg border border-dashed border-white/25 bg-[#262730]/60 hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[14px] text-[#fafafa] inline-flex items-center justify-center gap-2 transition-colors whitespace-nowrap cursor-pointer"
                >
                  <Upload className="w-4 h-4 shrink-0" />
                  <span>Import Settings (.json)</span>
                </button>
              </div>
            </div>
          </Expander>

          {pipelineState.running && (
            <div className="rounded-lg border border-white/16 bg-[#161922] p-4 space-y-2">
              <div className="flex items-center justify-between gap-3 text-[13.5px]">
                <span className="inline-flex items-center gap-2 font-medium text-[#fafafa]">
                  <Loader2 className="w-4 h-4 animate-spin text-[#ff4b4b]" />
                  <span>{pipelineState.stepLabel}</span>
                </span>
                <span className="font-mono tabular-nums text-[#fafafa]/75">
                  {pipelineState.progress}%
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                  style={{ width: `${pipelineState.progress}%` }}
                  className="h-full bg-[#ff4b4b] transition-all duration-300"
                />
              </div>
            </div>
          )}

          <button
            type="button"
            disabled={pipelineState.running}
            onClick={handleGenerateVideo}
            className="w-full min-h-[44px] px-6 py-2.5 rounded-lg bg-[#ff4b4b] hover:bg-[#e53e3e] active:bg-[#d43535] disabled:opacity-60 text-white text-[15px] font-medium inline-flex items-center justify-center gap-2 transition-colors whitespace-nowrap cursor-pointer"
          >
            {pipelineState.running ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Generating Video with Pollinations AI...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Generate Video</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Library Modal */}
      <LibraryModal
        open={libraryOpen}
        videos={libraryVideos}
        initialSelectedId={selectedLibraryVideoId}
        onDeleteVideo={handleDeleteLibraryVideo}
        onClose={() => setLibraryOpen(false)}
      />

      {/* Settings Modal */}
      <SettingsModal
        open={settingsOpen}
        autoOpenLibrary={autoOpenLibrary}
        onChangeAutoOpenLibrary={setAutoOpenLibrary}
        defaultVoice={settings.voiceName}
        onChangeDefaultVoice={(val) => updateField("voiceName", val)}
        defaultAspectRatio={settings.videoAspectRatio}
        onChangeDefaultAspectRatio={(val) => updateField("videoAspectRatio", val)}
        onClose={() => setSettingsOpen(false)}
      />

      {/* Final Compiled Prompt Preview */}
      <PromptPreviewModal
        open={promptPreviewOpen}
        systemPrompt={settings.systemPrompt}
        subject={settings.videoSubject}
        language={settings.scriptLanguage}
        paragraphCount={settings.paragraphCount}
        customRequirements={settings.customScriptRequirements}
        onClose={() => setPromptPreviewOpen(false)}
      />
    </div>
  );
}
