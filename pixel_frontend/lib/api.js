/*
  ============================================================
  lib/api.js  — talks to the FastAPI backend
  ============================================================
  Your backend (routes.py) has:

    POST /analyze   → upload an image or video (form field name: "file")
    GET  /health    → { "status": "ok" }

  Change the URL in .env.local:
    NEXT_PUBLIC_API_URL=http://localhost:8000
*/

// Base URL of FastAPI. Falls back to localhost:8000 if env is missing.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Sends the chosen file to POST /analyze
 * @param {File} file - image or video from the <input>
 * @param {(percent: number) => void} onProgress - called with 0–100 while waiting
 */
export function analyzeFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    // FormData is the browser way to send a file (same as HTML <form enctype="multipart/form-data">)
    const formData = new FormData();
    // The key MUST be "file" because FastAPI expects: file: UploadFile = File(...)
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/analyze`);

    let fake = 8;
    // The backend does not stream 0–100%. This timer only updates the UI while we wait.
    const waitTimer = setInterval(() => {
      fake = Math.min(90, fake + 4);
      onProgress(fake);
    }, 400);

    function stopTimer() {
      clearInterval(waitTimer);
    }

    // Upload progress (0–70%). The remaining time is "server is analysing".
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const uploaded = Math.round((event.loaded / event.total) * 70);
      fake = Math.max(fake, uploaded);
      onProgress(fake);
    };

    xhr.onload = () => {
      stopTimer();
      onProgress(100);
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch {
        data = { detail: xhr.responseText };
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
        return;
      }

      const message = data.detail || `Request failed (${xhr.status})`;
      reject(new Error(message));
    };

    xhr.onerror = () => {
      stopTimer();
      reject(
        new Error(
          "Could not reach the API. Is FastAPI running, and is NEXT_PUBLIC_API_URL correct?"
        )
      );
    };

    onProgress(2);
    xhr.send(formData);
  });
}

/** Optional check that the backend is alive: GET /health */
export async function checkHealth() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export { API_URL };
