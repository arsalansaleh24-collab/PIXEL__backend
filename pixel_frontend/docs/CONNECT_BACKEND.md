# How to connect this frontend to your FastAPI backend

Your backend already has CORS enabled (`allow_origins=["*"]`), so the browser is allowed to call it during development.

## 1. What the backend expects

From your `routes.py`:

| Method | Path | Body | Notes |
|--------|------|------|--------|
| POST | `/analyze` | multipart form, field name **`file`** | Image or video. Other types → 400 `bad file type` |
| GET | `/health` | none | `{ "status": "ok" }` |

The frontend already uses the field name `file`. Do **not** rename it unless you also change FastAPI.

## 2. Start FastAPI

From your backend folder (the one that contains `main.py`):

```bash
uvicorn main:app --reload --port 8000
```

Check it works:

```bash
curl http://localhost:8000/health
```

You should see `{"status":"ok"}`.

## 3. Point Next.js at that URL

In the frontend folder:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_GITHUB_URL=https://github.com/YOUR_USERNAME/your-repo
```

Restart `npm run dev` after changing env files. Next.js reads `NEXT_PUBLIC_*` at startup.

The actual request is built in `lib/api.js`:

```
POST http://localhost:8000/analyze
Content-Type: multipart/form-data
file = <the image or video>
```

## 4. What comes back (response)

This frontend does **not** guess a special shape. It shows the JSON that `build_resp(res, is_vid)` returns.

If your JSON looks like this:

```json
{
  "label": "fake",
  "confidence": 0.91
}
```

that exact object appears on the result screen.

If you later want a nicer card (big “REAL / FAKE” label), tell me the exact keys from `build_resp` and we can map them.

## 5. Roadmap (when to wire what)

| Step | Frontend | Backend | Done in this repo? |
|------|----------|---------|--------------------|
| A | Upload UI + progress bar | — | Yes (`UploadAnalyser.js`) |
| B | `lib/api.js` POST `/analyze` | `routes.py` `analyze()` | Yes — works as soon as FastAPI is running |
| C | Show JSON result | `build_resp(...)` | Yes — generic JSON display |
| D | Pretty verdict UI | Share the JSON keys | Not yet — waiting on your response shape |
| E | Restrict CORS in production | Change `allow_origins=["*"]` to your real site URL | Not yet (you already marked this TODO) |
| F | Deploy frontend + backend | Same `NEXT_PUBLIC_API_URL` as the public API | Not yet |

## 6. Common errors

| What you see | Likely cause |
|--------------|----------------|
| “Could not reach the API” | FastAPI is not running, or the port is not 8000 |
| `bad file type` | MIME type is not in `img_types` / `vid_types` in `api/config.py` |
| CORS error in the browser console | Frontend origin not allowed (only an issue after you tighten CORS) |
| 500 from `/analyze` | Exception inside `analyze_image` / `analyze_video` |

## 7. Production note

When you deploy:

1. Put the real API URL in `NEXT_PUBLIC_API_URL` (example: `https://api.yourproject.com`).
2. In FastAPI, replace `allow_origins=["*"]` with your website origin, for example `https://your-frontend.vercel.app`.
