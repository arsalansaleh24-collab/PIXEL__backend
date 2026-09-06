"use client";

/*
  Main interactive area:
  1) idle     → drag & drop / click / Upload File button
  2) analyzing → progress bar 0% to 100%
  3) result   → show backend JSON in a readable way
  4) error    → show the error and let the user try again
*/
import { useRef, useState } from "react";
import { analyzeFile } from "@/lib/api";

const ACCEPT = "image/*,video/*";

function isAllowedFile(file) {
  if (!file) return false;
  return file.type.startsWith("image/") || file.type.startsWith("video/");
}

export default function UploadAnalyser() {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | analyzing | done | error
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

function IdleState({ inputRef, onPick, onFile }) {
  return (
    <div className="flex flex-col items-center text-center">
      {/* Hidden file picker — opened by clicking the card or the button */}
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
  return (
    <div className="mx-auto w-full max-w-lg text-left">
      <p className="text-sm text-[var(--muted)]">Result</p>
      <p className="mt-1 truncate text-lg font-medium">{fileName}</p>
      <pre className="mt-6 overflow-x-auto rounded-2xl border border-[var(--border)] bg-[var(--background)] p-4 text-sm leading-6">
        {JSON.stringify(result, null, 2)}
      </pre>
      <button
        type="button"
        onClick={onReset}
        className="mt-6 inline-flex rounded-full border border-[var(--border)] px-5 py-2 text-sm font-medium hover:bg-[var(--soft-blue)]"
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
