export interface DashboardSettings {
  // Script Settings (Gemini API)
  videoSubject: string;
  scriptLanguage: string;
  paragraphCount: number;
  customScriptRequirements: string;
  systemPrompt: string;
  videoScript: string;
  videoKeywords: string;

  // Video Settings (Pollinations AI)
  videoSource: "Pollinations AI";
  pollinationsVideoModel: string;
  videoConcatMode: string;
  matchMaterialsToScript: boolean;
  photoAnimationMode: string;
  videoTransitionMode: string;
  videoAspectRatio: string;
  sourceFrameFit: string;
  maxClipDuration: string;
  clipSpeed: number;
  videosPerRun: string;
  videoEncoder: string;
  clipRenderingConcurrency: string;

  // Audio Settings (Gemini TTS)
  voiceoverMode: "Auto" | "Upload" | "None";
  ttsService: "Google Gemini TTS";
  voiceName: string;
  voiceVolume: string;
  voiceSpeed: string;
  bgmSource: string;
  bgmVolume: string;

  // Subtitle Settings
  subtitleEnabled: boolean;
  subtitleUppercase: boolean;
  fontName: string;
  subtitlePosition: string;
  customPositionPercent: number;
  subtitleDisplayMode: string;
  entranceAnimation: string;
  fontColor: string;
  fontSize: number;
  strokeColor: string;
  strokeWidth: number;
  textShadowEnabled: boolean;
  textShadowColor: string;
  textShadowOffset: number;
  textShadowBlur: number;
  textShadowOpacity: number;
  subtitleBackgroundEnabled: boolean;
  subtitleBackgroundColor: string;
  roundedSubtitleBackground: boolean;
}

export interface LibraryVideo {
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

export const POLLINATIONS_VIDEO_MODELS = [
  "wan-fast (Alibaba Wan 2.2 Fast)",
  "seedance-pro (ByteDance Seedance Pro)",
  "seedance-2.0-fast (ByteDance Seedance 2.0 Fast)",
  "p-video (Pruna P-Video)",
  "wan (Alibaba Wan 2.6)",
  "veo (Google Veo 3.1 Fast)",
];

export const PHOTO_ANIMATION_MODES = [
  "Auto Dynamic (Zoom + Slide + Pan)",
  "Zoom In",
  "Zoom Out",
  "Alternate Zoom (In & Out)",
  "Slide Left & Right",
  "Slide Left",
  "Slide Right",
  "Slide Up & Down",
  "Ken Burns Drift (Pan + Zoom)",
  "None (Static Photo)",
];

export const VIDEO_TRANSITION_MODES = [
  "Shuffle",
  "ZoomIn",
  "ZoomOut",
  "SlideIn",
  "SlideOut",
  "FadeIn",
  "FadeOut",
  "None",
];

export const DEFAULT_SYSTEM_PROMPT = `# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.`;

export const DEFAULT_SUBTITLE_SETTINGS = {
  subtitleEnabled: true,
  subtitleUppercase: true,
  fontName: "UTM Kabel KT.ttf",
  subtitlePosition: "Bottom (Recommended)",
  customPositionPercent: 80,
  subtitleDisplayMode: "Whole sentence",
  entranceAnimation: "Cinema Clean Dissolve",
  fontColor: "#FFFFFF",
  fontSize: 56,
  strokeColor: "#000000",
  strokeWidth: 1.5,
  textShadowEnabled: true,
  textShadowColor: "#000000",
  textShadowOffset: 4,
  textShadowBlur: 2,
  textShadowOpacity: 80,
  subtitleBackgroundEnabled: false,
  subtitleBackgroundColor: "#000000",
  roundedSubtitleBackground: false,
};

export const DEFAULT_DASHBOARD_SETTINGS: DashboardSettings = {
  videoSubject: "",
  scriptLanguage: "en-US",
  paragraphCount: 1,
  customScriptRequirements: "",
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  videoScript: "",
  videoKeywords: "",

  videoSource: "Pollinations AI",
  pollinationsVideoModel: "wan-fast (Alibaba Wan 2.2 Fast)",
  videoConcatMode: "Sequential Concatenation",
  matchMaterialsToScript: true,
  photoAnimationMode: "Auto Dynamic (Zoom + Slide + Pan)",
  videoTransitionMode: "Shuffle",
  videoAspectRatio: "Portrait 9:16",
  sourceFrameFit: "Fill (center crop)",
  maxClipDuration: "4",
  clipSpeed: 1.0,
  videosPerRun: "1",
  videoEncoder: "Default (Recommended)",
  clipRenderingConcurrency: "2",

  voiceoverMode: "Auto",
  ttsService: "Google Gemini TTS",
  voiceName: "gemini:Puck-Upbeat",
  voiceVolume: "100%",
  voiceSpeed: "1.0",
  bgmSource: "Random Background Music",
  bgmVolume: "20%",

  ...DEFAULT_SUBTITLE_SETTINGS,
};

export const GEMINI_TTS_VOICES = [
  "gemini:Zephyr-Bright",
  "gemini:Puck-Upbeat",
  "gemini:Charon-Informative",
  "gemini:Kore-Firm",
  "gemini:Fenrir-Excitable",
  "gemini:Leda-Youthful",
  "gemini:Orus-Firm",
  "gemini:Aoede-Breezy",
  "gemini:Callirrhoe-Easy-going",
  "gemini:Autonoe-Bright",
  "gemini:Enceladus-Breathy",
  "gemini:Iapetus-Clear",
  "gemini:Umbriel-Easy-going",
  "gemini:Algieba-Smooth",
  "gemini:Despina-Smooth",
  "gemini:Erinome-Clear",
  "gemini:Algenib-Gravelly",
  "gemini:Rasalgethi-Informative",
  "gemini:Laomedeia-Upbeat",
  "gemini:Achernar-Soft",
  "gemini:Alnilam-Firm",
  "gemini:Schedar-Even",
  "gemini:Gacrux-Mature",
  "gemini:Pulcherrima-Forward",
  "gemini:Achird-Friendly",
  "gemini:Zubenelgenubi-Casual",
  "gemini:Vindemiatrix-Gentle",
  "gemini:Sadachbia-Lively",
  "gemini:Sadaltager-Knowledgeable",
  "gemini:Sulafat-Warm",
];

export const SCRIPT_LANGUAGES = [
  "Auto Detect",
  "ca-ES",
  "zh-CN",
  "zh-HK",
  "zh-TW",
  "de-DE",
  "en-US",
  "es-ES",
  "fr-FR",
  "it-IT",
  "ru-RU",
  "vi-VN",
  "th-TH",
  "tr-TR",
];

export const SUBTITLE_FONTS = [
  "BeVietnamPro-Bold.ttf",
  "BeVietnamPro-Medium.ttf",
  "Charm-Bold.ttf",
  "Charm-Regular.ttf",
  "MicrosoftYaHeiBold.ttc",
  "MicrosoftYaHeiNormal.ttc",
  "STHeitiLight.ttc",
  "STHeitiMedium.ttc",
  "UTM Kabel KT.ttf",
];
