import React, { useState, useEffect } from "react";
import { X, Download, Trash2, Film, Play } from "lucide-react";
import type { LibraryVideo } from "../types/dashboard";

interface LibraryModalProps {
  open: boolean;
  videos: LibraryVideo[];
  initialSelectedId?: string | null;
  onDeleteVideo: (id: string) => void;
  onClose: () => void;
}

export const LibraryModal: React.FC<LibraryModalProps> = ({
  open,
  videos,
  initialSelectedId,
  onDeleteVideo,
  onClose,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    if (initialSelectedId && videos.some((v) => v.id === initialSelectedId)) {
      setSelectedId(initialSelectedId);
    } else if (videos.length > 0) {
      setSelectedId(videos[0].id);
    } else {
      setSelectedId(null);
    }
  }, [open, initialSelectedId, videos]);

  if (!open) return null;

  const activeVideo =
    videos.find((v) => v.id === selectedId) || videos[0] || null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Video Library"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs"
    >
      <div className="w-full max-w-5xl max-h-[88vh] flex flex-col rounded-xl bg-[#161922] border border-white/16 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between gap-4 px-5 py-3.5 border-b border-white/12 shrink-0">
          <div className="flex items-center gap-2.5">
            <Film className="w-5 h-5 text-[#ff4b4b]" />
            <h2 className="text-[16px] font-semibold text-[#fafafa]">
              Library
            </h2>
            <span className="text-[12.5px] font-mono tabular-nums text-[#fafafa]/55">
              · {videos.length} {videos.length === 1 ? "video" : "videos"}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close library"
            className="p-1.5 rounded-lg text-[#fafafa]/65 hover:text-[#fafafa] hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        {videos.length === 0 ? (
          <div className="p-12 flex flex-col items-center justify-center text-center space-y-3">
            <div className="w-12 h-12 rounded-xl bg-[#262730] border border-white/10 flex items-center justify-center text-[#fafafa]/50">
              <Film className="w-6 h-6" />
            </div>
            <h3 className="text-[15px] font-medium text-[#fafafa]">
              Your generated video library is empty
            </h3>
            <p className="text-[13.5px] text-[#fafafa]/60 max-w-md">
              Enter a Video Subject on the main dashboard and click Generate Video. Pollinations AI scenes and Gemini voiceovers will be rendered into downloadable MP4 files here.
            </p>
            <button
              type="button"
              onClick={onClose}
              className="mt-2 px-4 py-2 rounded-lg bg-[#ff4b4b] hover:bg-[#e53e3e] text-white text-[13.5px] font-medium transition-colors cursor-pointer whitespace-nowrap"
            >
              Create First Video
            </button>
          </div>
        ) : (
          <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[320px_1fr] divide-y lg:divide-y-0 lg:divide-x divide-white/12 overflow-hidden">
            {/* Left Sidebar: Generated Video List */}
            <div className="overflow-y-auto divide-y divide-white/8 max-h-[260px] lg:max-h-none">
              {videos.map((vid) => {
                const isSelected = activeVideo?.id === vid.id;
                return (
                  <div
                    key={vid.id}
                    onClick={() => setSelectedId(vid.id)}
                    className={`p-3.5 flex items-start justify-between gap-3 cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-white/[0.08]"
                        : "hover:bg-white/[0.04]"
                    }`}
                  >
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center gap-1.5">
                        <Play
                          className={`w-3.5 h-3.5 shrink-0 ${
                            isSelected ? "text-[#ff4b4b] fill-current" : "text-[#fafafa]/50"
                          }`}
                        />
                        <span className="text-[13.5px] font-medium text-[#fafafa] truncate">
                          {vid.subject}
                        </span>
                      </div>
                      <div className="text-[12px] text-[#fafafa]/55 font-mono tabular-nums">
                        <span>{vid.aspectRatio}</span>
                        <span aria-hidden="true"> · </span>
                        <span>{vid.durationSeconds}s</span>
                        <span aria-hidden="true"> · </span>
                        <span>{vid.createdAt}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1 shrink-0">
                      <a
                        href={vid.downloadUrl}
                        download={vid.filename}
                        onClick={(e) => e.stopPropagation()}
                        title="Download MP4"
                        aria-label={`Download ${vid.subject}`}
                        className="p-1.5 rounded-md border border-white/15 bg-[#262730] hover:border-[#ff4b4b] hover:text-[#ff4b4b] text-[#fafafa] transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </a>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteVideo(vid.id);
                        }}
                        title="Delete Video"
                        aria-label={`Delete ${vid.subject}`}
                        className="p-1.5 rounded-md border border-white/15 bg-[#262730] hover:border-[#dc2626] hover:text-[#f87171] text-[#fafafa]/80 transition-colors cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Right Pane: Active Video Player & Download */}
            {activeVideo && (
              <div className="p-5 overflow-y-auto flex flex-col md:flex-row gap-5 items-start">
                <div className="w-full md:w-auto shrink-0 flex justify-center mx-auto">
                  <video
                    key={activeVideo.videoUrl}
                    src={activeVideo.videoUrl}
                    controls
                    autoPlay
                    playsInline
                    className={`rounded-lg bg-black border border-white/15 shadow-lg ${
                      activeVideo.aspectRatio.includes("16:9")
                        ? "w-full md:w-[420px] aspect-video"
                        : activeVideo.aspectRatio.includes("1:1")
                        ? "w-[280px] aspect-square"
                        : "w-[240px] aspect-[9/16]"
                    }`}
                  />
                </div>

                <div className="flex-1 min-w-0 space-y-4 w-full">
                  <div>
                    <h3 className="text-[17px] font-semibold text-[#fafafa]">
                      {activeVideo.subject}
                    </h3>
                    <p className="text-[12.5px] text-[#fafafa]/60 font-mono tabular-nums mt-0.5">
                      <span>{activeVideo.videoSource}</span>
                      <span aria-hidden="true"> · </span>
                      <span>{activeVideo.aspectRatio}</span>
                      <span aria-hidden="true"> · </span>
                      <span>{activeVideo.durationSeconds}s</span>
                      <span aria-hidden="true"> · </span>
                      <span>{activeVideo.createdAt}</span>
                    </p>
                  </div>

                  <div>
                    <div className="text-[12.5px] font-semibold text-[#fafafa]/70 mb-1.5">
                      Generated Script
                    </div>
                    <div className="p-3 rounded-lg bg-[#262730] border border-white/10 text-[13.5px] leading-relaxed text-[#fafafa]/90 max-h-40 overflow-y-auto">
                      {activeVideo.script}
                    </div>
                  </div>

                  {activeVideo.keywords.length > 0 && (
                    <div>
                      <div className="text-[12.5px] font-semibold text-[#fafafa]/70 mb-1">
                        Pollinations AI Scene Prompts
                      </div>
                      <p className="text-[13px] text-[#fafafa]/75">
                        {activeVideo.keywords.join(" · ")}
                      </p>
                    </div>
                  )}

                  <div className="pt-2 flex flex-wrap items-center gap-3">
                    <a
                      href={activeVideo.downloadUrl}
                      download={activeVideo.filename}
                      className="px-4 py-2.5 rounded-lg bg-[#ff4b4b] hover:bg-[#e53e3e] text-white text-[14px] font-medium inline-flex items-center gap-2 transition-colors whitespace-nowrap shrink-0"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download Video (.mp4)</span>
                    </a>

                    <button
                      type="button"
                      onClick={() => onDeleteVideo(activeVideo.id)}
                      className="px-3.5 py-2.5 rounded-lg border border-white/18 hover:border-[#dc2626] hover:text-[#f87171] text-[13.5px] text-[#fafafa]/85 inline-flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
