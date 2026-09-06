const GITHUB_URL =
  process.env.NEXT_PUBLIC_GITHUB_URL || "https://github.com/YOUR_USERNAME/pblpixel_frontend";

export default function Footer() {
  return (
    <footer className="border-t border-[var(--border)]">
      <div className="mx-auto flex w-full max-w-3xl flex-col items-center gap-4 px-4 py-8 text-center text-sm text-[var(--muted)] sm:flex-row sm:justify-between sm:text-left">
        <p>PIXEL · SMIT CSE PBL · 2024–2028</p>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          className="underline decoration-[var(--border)] underline-offset-4 hover:text-[var(--primary)]"
        >
          View Repository on GitHub
        </a>
      </div>
    </footer>
  );
}
