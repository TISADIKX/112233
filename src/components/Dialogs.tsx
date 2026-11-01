import React from "react";
import { X, Save } from "lucide-react";
import { CheckboxField, SelectField } from "./FormPrimitives";
import { GEMINI_TTS_VOICES } from "../types/dashboard";

interface SettingsModalProps {
  open: boolean;
  autoOpenLibrary: boolean;
  onChangeAutoOpenLibrary: (val: boolean) => void;
  defaultVoice: string;
  onChangeDefaultVoice: (val: string) => void;
  defaultAspectRatio: string;
  onChangeDefaultAspectRatio: (val: string) => void;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  autoOpenLibrary,
  onChangeAutoOpenLibrary,
  defaultVoice,
  onChangeDefaultVoice,
  defaultAspectRatio,
  onChangeDefaultAspectRatio,
  onClose,
}) => {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs"
    >
      <div className="w-full max-w-lg rounded-xl bg-[#161922] border border-white/16 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/12">
          <h2 className="text-[16px] font-semibold text-[#fafafa]">
            Settings
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="p-1.5 rounded-lg text-[#fafafa]/65 hover:text-[#fafafa] hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <SelectField
            label="Default Gemini TTS Voice"
            value={defaultVoice}
            options={GEMINI_TTS_VOICES}
            onChange={onChangeDefaultVoice}
          />

          <SelectField
            label="Default Video Aspect Ratio"
            value={defaultAspectRatio}
            options={["Portrait 9:16", "Landscape 16:9", "Square 1:1"]}
            onChange={onChangeDefaultAspectRatio}
          />

          <CheckboxField
            label="Automatically open Library after video generation finishes"
            checked={autoOpenLibrary}
            onChange={onChangeAutoOpenLibrary}
          />

          <div className="flex justify-end pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-[#ff4b4b] hover:bg-[#e53e3e] text-white text-[13.5px] font-medium inline-flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap"
            >
              <Save className="w-4 h-4" />
              <span>Save</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

interface PromptPreviewModalProps {
  open: boolean;
  systemPrompt: string;
  subject: string;
  language: string;
  paragraphCount: number;
  customRequirements: string;
  onClose: () => void;
}

export const PromptPreviewModal: React.FC<PromptPreviewModalProps> = ({
  open,
  systemPrompt,
  subject,
  language,
  paragraphCount,
  customRequirements,
  onClose,
}) => {
  if (!open) return null;

  const finalCompiled = `${systemPrompt}\n\n---\n[User Request Payload]\nVideo Subject: ${
    subject.trim() || "(Example: How AI is changing everyday life)"
  }\nScript Language: ${language}\nParagraph Count: ${paragraphCount}${
    customRequirements.trim()
      ? `\nCustom Script Requirements: ${customRequirements.trim()}`
      : ""
  }`;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Preview Final Prompt"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs"
    >
      <div className="w-full max-w-2xl rounded-xl bg-[#161922] border border-white/16 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/12">
          <h2 className="text-[16px] font-semibold text-[#fafafa]">
            Final Compiled Script Prompt
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close prompt preview"
            className="p-1.5 rounded-lg text-[#fafafa]/65 hover:text-[#fafafa] hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-5">
          <pre className="w-full max-h-[60vh] overflow-y-auto p-3.5 rounded-lg bg-[#262730] font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap text-[#fafafa]">
            {finalCompiled}
          </pre>
        </div>
        <div className="px-5 py-3 bg-white/[0.03] border-t border-white/10 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-white/18 hover:bg-white/10 text-[13.5px] text-[#fafafa] transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
