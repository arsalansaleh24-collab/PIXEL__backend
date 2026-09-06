# How this frontend is organised

This folder is a map of the PIXEL website. Read this if you are new to Next.js.

PIXEL is a **single page**. There is no extra login, pricing, or demo section.

---

## Folders at a glance

```
pblpixel_frontend/
├── app/                 ← Next.js pages live here
│   ├── layout.js        ← wraps every page (title, font, theme script)
│   ├── page.js          ← THE only page: header + upload + about + faqs + footer
│   └── globals.css      ← colors and Tailwind
├── components/          ← reusable pieces of the UI (buttons, sections)
│   ├── Header.js        ← centered P.I.X.E.L title
│   ├── ThemeToggle.js   ← moon/sun button (top-right)
│   ├── UploadAnalyser.js← drag & drop, upload button, progress, result
│   ├── AboutUs.js       ← About section
│   ├── Faqs.js          ← 3 FAQs
│   └── Footer.js        ← GitHub link
├── lib/
│   └── api.js           ← JavaScript that calls FastAPI (/analyze, /health)
├── docs/                ← you are here (guides, not the website)
├── .env.example         ← copy this to .env.local
└── package.json         ← npm scripts (dev, build)
```

`node_modules/` is installed packages. You do not edit it.

---

## What each file does

| File | Job |
|------|-----|
| `app/page.js` | Assembles the page. No buttons here — it only imports components. |
| `app/layout.js` | Sets the browser tab title to P.I.X.E.L and loads CSS/font. |
| `app/globals.css` | Light/dark colors. Tailwind classes (`flex`, `rounded-full`, …) come from here. |
| `components/Header.js` | Shows **P.I.X.E.L** in the center. |
| `components/ThemeToggle.js` | **Theme button** — adds/removes the `dark` class on `<html>`. Saves choice in `localStorage`. |
| `components/UploadAnalyser.js` | **Upload File button**, hidden file input, drag-and-drop zone, 0–100% bar, result/error. |
| `components/AboutUs.js` | About Us text (SMIT CSE PBL). |
| `components/Faqs.js` | Three FAQ questions. |
| `components/Footer.js` | **View Repository on GitHub** link. |
| `lib/api.js` | `analyzeFile()` posts the file to FastAPI. |

---

## Every button / clickable thing

| What you click | File | What it does |
|----------------|------|----------------|
| Moon / sun icon | `ThemeToggle.js` | Light ↔ dark mode |
| Whole upload card | `UploadAnalyser.js` (`IdleState`) | Opens the file picker |
| **Upload File** | `UploadAnalyser.js` | Same file picker (images + videos) |
| **Analyse another file** | `UploadAnalyser.js` (`ResultState`) | Goes back to the empty upload card |
| **Try again** | `UploadAnalyser.js` (`ErrorState`) | Clears the error |
| **View Repository on GitHub** | `Footer.js` | Opens the repo URL from `.env.local` |

Drag-and-drop is not a button. It is handled in `UploadAnalyser.js` with `onDrop`.

There is **no** “Get a Free Demo”, contact form, or example-model tags.

---

## How a file moves through the app

1. User drops a file or clicks **Upload File**.
2. `handleFile()` in `UploadAnalyser.js` checks it is an image or video.
3. Status becomes `analyzing` and the progress bar is shown.
4. `analyzeFile()` in `lib/api.js` sends `FormData` with the field name **`file`** to `POST {API}/analyze`.
5. FastAPI (`routes.py`) runs `analyze_image` or `analyze_video` and returns JSON.
6. The JSON is printed on the page. If the server is offline, **Try again** is shown.

---

## Run the site

```bash
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000

Start FastAPI separately (often http://localhost:8000). See `CONNECT_BACKEND.md`.
