"use client";

/*
  Main interactive area:
  1) idle       → drag & drop / click / Upload File button
  2) analyzing  → progress bar 0% to 100%
  3) result     → visual results UI with full metrics from utils.py
  4) error      → error display with retry
*/
import React, { useRef, useState } from "react";
import { analyzeFile } from "@/lib/api";

const ACCEPT = "image/*,video/*";

function isAllowedFile(file) {
    if (!file) return false;
    return file.type.startsWith("image/") || file.type.startsWith("video/");
}

export function UploadAnalyser() {
    const inputRef = useRef(null);
    const [isDragging, setIsDragging] = useState(false);
    const [status, setStatus] = useState("idle");
    const [progress, setProgress] = useState(0);
    const [fileName, setFileName] = useState("");
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    function reset() {
        setStatus("idle");
        setProgress(0);
        setFileName("");
        setResult(null);
        setError("");
        if (inputRef.current) inputRef.current.value = "";
    }

    async function handleFile(file) {
        if (!isAllowedFile(file)) {
            setStatus("error");
            setError("Please choose an image or video file.");
            return;
        }

        setFileName(file.name);
        setStatus("analyzing");
        setProgress(0);
        setError("");
        setResult(null);

        try {
            const data = await analyzeFile(file, setProgress);
            setResult(data);
            setProgress(100);
            setStatus("done");
        } catch (err) {
            setStatus("error");
            setError(err.message || "Analysis failed.");
        }
    }

    function onDrop(event) {
        event.preventDefault();
        setIsDragging(false);
        const file = event.dataTransfer.files?.[0];
        if (file) handleFile(file);
    }

    return (
        <section className="mx-auto w-full max-w-3xl px-4">
            <div
                className={`rounded-3xl border bg-[var(--card)] px-6 py-12 shadow-sm transition sm:px-12 sm:py-16 ${
                    isDragging
                        ? "border-[var(--primary)] ring-2 ring-[var(--primary)]/30"
                        : "border-[var(--border)]"
                }`}
                onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
            >
                {status === "idle" && (
                    <IdleState
                        inputRef={inputRef}
                        onPick={() => inputRef.current?.click()}
                        onFile={(file) => handleFile(file)}
                    />
                )}

                {status === "analyzing" && (
                    <ProgressState fileName={fileName} progress={progress} />
                )}

                {status === "done" && (
                    <ResultState fileName={fileName} result={result} onReset={reset} />
                )}

                {status === "error" && (
                    <ErrorState message={error} onReset={reset} />
                )}
            </div>
        </section>
    );
}

export default UploadAnalyser;

function IdleState({ inputRef, onPick, onFile }) {
    return (
        <div className="flex flex-col items-center text-center">
            <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onFile(file);
                }}
            />

            <button
                type="button"
                onClick={onPick}
                className="flex w-full flex-col items-center"
                aria-label="Choose an image or video to upload"
            >
                <UploadIcon />
                <h2 className="mt-6 max-w-lg text-2xl font-semibold text-[var(--primary)] sm:text-3xl dark:text-blue-200">
                    Drag & Drop or Click to Upload an Image or Video
                </h2>
                <p className="mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">
                    PIXEL accepts image and video files. Drop a file here or use the button below.
                </p>
            </button>

            <button
                type="button"
                onClick={onPick}
                className="mt-8 inline-flex items-center gap-2 rounded-full bg-[#1e3a8a] px-8 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-[#1e40af] dark:bg-blue-400 dark:text-slate-950 dark:hover:bg-blue-300"
            >
                <CloudIcon />
                Upload File
            </button>
        </div>
    );
}

function ProgressState({ fileName, progress }) {
    const value = Math.min(100, Math.max(0, Math.round(progress)));

    return (
        <div className="mx-auto w-full max-w-lg text-center">
            <p className="text-sm text-[var(--muted)]">Analysing</p>
            <p className="mt-1 truncate text-lg font-medium">{fileName}</p>
            <div className="mt-8 h-3 overflow-hidden rounded-full bg-[var(--soft-blue)]">
                <div
                    className="h-full rounded-full bg-[#1e3a8a] transition-all duration-200 dark:bg-blue-400"
                    style={{ width: `${value}%` }}
                />
            </div>
            <p className="mt-3 text-2xl font-semibold tabular-nums text-[var(--primary)] dark:text-blue-200">
                {value}%
            </p>
            <p className="mt-2 text-sm text-[var(--muted)]">
                Please wait while PIXEL processes your file.
            </p>
        </div>
    );
}

function ResultState({ fileName, result, onReset }) {
    const status = result?.status || "ANALYSIS COMPLETE";
    const classification = result?.classification || "UNKNOWN";
    const confidenceScore = result?.confidence_score || "0%";

    const details = result?.details || {};
    const message = details.message || "None";
    const detector = details.detector || "Unknown";
    const faceDetected = details.face_detected || "No";
    const framesAnalyzed = details.frames_analyzed;
    const temporalConsistency = details.temporal_consistency;

    const numericScore = Math.min(100, Math.max(0, parseFloat(confidenceScore.replace("%", "")) || 0));
    const isDeepfake = classification.toUpperCase() === "DEEPFAKE";
    const authenticityPct = isDeepfake ? Math.max(0, 100 - numericScore) : numericScore;

    let barColor = "#22c55e"; // Green (>80%)
    let glowColor = "rgba(34, 197, 94, 0.45)";

    if (authenticityPct < 20) {
        barColor = "#ef4444"; // Red (<20%)
        glowColor = "rgba(239, 68, 68, 0.45)";
    } else if (authenticityPct <= 80) {
        barColor = "#f97316"; // Orange (20-80%)
        glowColor = "rgba(249, 115, 22, 0.45)";
    }

    return (
        <div className="mx-auto w-full max-w-xl text-left font-sans">
            <h2 className="text-2xl font-bold tracking-tight text-slate-100">Results</h2>
            <p className="mt-1 text-sm leading-relaxed text-slate-400">
                Authenticity score: likelihood the face is real. Red &lt;20%, Orange 20–80%, Green &gt;80%.
            </p>

            <p className="mt-4 truncate text-xs font-semibold tracking-wider text-slate-400 uppercase">
                File: <span className="text-slate-200">{fileName}</span>
            </p>

            {/* Authenticity Bar Track */}
            <div className="mt-3">
                <div className="h-3.5 w-full overflow-hidden rounded-full bg-slate-950 p-[2px] ring-1 ring-slate-800">
                    <div
                        className="h-full rounded-full transition-all duration-700 ease-out"
                        style={{
                            width: `${authenticityPct}%`,
                            backgroundColor: barColor,
                            boxShadow: `0 0 12px ${glowColor}`
                        }}
                    />
                </div>

                <div className="mt-2.5 flex items-baseline justify-between">
          <span className="text-xl font-bold tracking-tight text-slate-100">
            {authenticityPct.toFixed(2)}% authentic
          </span>
                    <span className="text-xs font-semibold text-slate-400 tabular-nums">
            Raw Confidence: {confidenceScore}
          </span>
                </div>
            </div>

            {/* Metrics breakdown */}
            <div className="mt-6 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-slate-800/90 bg-slate-950/70 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              status
            </span>
                        <p className="mt-1 text-sm font-bold text-emerald-400">{status}</p>
                    </div>

                    <div className="rounded-xl border border-slate-800/90 bg-slate-950/70 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              classification
            </span>
                        <p
                            className="mt-1 text-sm font-bold uppercase tracking-wide"
                            style={{ color: barColor }}
                        >
                            {classification}
                        </p>
                    </div>
                </div>

                <div className="rounded-xl border border-slate-800/90 bg-slate-950/70 p-4">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            details
          </span>

                    <div className="mt-3 grid grid-cols-2 gap-y-3.5 gap-x-4 sm:grid-cols-3">
                        <div>
                            <p className="text-xs text-slate-400 font-medium">message</p>
                            <p className="mt-0.5 text-sm font-semibold text-slate-100">
                                &quot;{message}&quot;
                            </p>
                        </div>

                        <div>
                            <p className="text-xs text-slate-400 font-medium">detector</p>
                            <p className="mt-0.5 text-sm font-semibold text-slate-100">
                                {detector}
                            </p>
                        </div>

                        <div>
                            <p className="text-xs text-slate-400 font-medium">face_detected</p>
                            <p className="mt-0.5 text-sm font-semibold text-slate-100">
                                {faceDetected}
                            </p>
                        </div>

                        {framesAnalyzed !== undefined && (
                            <div>
                                <p className="text-xs text-slate-400 font-medium">frames_analyzed</p>
                                <p className="mt-0.5 text-sm font-semibold text-slate-100 tabular-nums">
                                    {framesAnalyzed}
                                </p>
                            </div>
                        )}

                        {temporalConsistency && (
                            <div>
                                <p className="text-xs text-slate-400 font-medium">temporal_consistency</p>
                                <p
                                    className={`mt-0.5 text-sm font-semibold ${
                                        temporalConsistency === "Suspicious" ? "text-red-400" : "text-emerald-400"
                                    }`}
                                >
                                    {temporalConsistency}
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <button
                type="button"
                onClick={onReset}
                className="mt-6 inline-flex rounded-full border border-slate-700 bg-slate-900 px-5 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800 hover:text-white"
            >
                Analyse another file
            </button>
        </div>
    );
}

function ErrorState({ message, onReset }) {
    return (
        <div className="mx-auto max-w-md text-center">
            <p className="text-lg font-medium text-red-700 dark:text-red-300">Could not analyse this file</p>
            <p className="mt-2 text-sm text-[var(--muted)]">{message}</p>
            <button
                type="button"
                onClick={onReset}
                className="mt-6 inline-flex rounded-full bg-[#1e3a8a] px-5 py-2 text-sm font-medium text-white dark:bg-blue-400 dark:text-slate-950"
            >
                Try again
            </button>
        </div>
    );
}

function UploadIcon() {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" className="h-16 w-16 text-[#1e3a8a] dark:text-blue-300" fill="none" stroke="currentColor" strokeWidth="2.2">
            <rect x="10" y="14" width="44" height="36" rx="4" />
            <path d="M10 42l12-12 8 8 6-6 18 16" />
            <circle cx="22" cy="26" r="3" />
            <circle cx="48" cy="12" r="8" fill="#1e3a8a" stroke="none" className="dark:fill-blue-300" />
            <path d="M48 8v8M44 12h8" stroke="white" strokeWidth="2" />
        </svg>
    );
}

function CloudIcon() {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
            <path d="M16 16h1.5A3.5 3.5 0 0 0 21 12.5 3.5 3.5 0 0 0 17.6 9a5 5 0 0 0-9.5 1.2A3.5 3.5 0 0 0 5 16.5H8" />
            <path d="M12 12v8M9 15l3-3 3 3" />
        </svg>
    );
}
