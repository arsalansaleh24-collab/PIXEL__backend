// Header is a Server Component (no "use client") except for ThemeToggle inside it.
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  return (
    <header className="relative mx-auto flex w-full max-w-5xl items-center justify-center px-4 py-8 sm:py-10">
      {/* Title stays in the visual center of the page */}
      <h1 className="text-center text-3xl font-semibold tracking-[0.28em] text-[var(--primary)] sm:text-4xl dark:text-blue-200">
        P.I.X.E.L
      </h1>

      {/* Theme button sits in the top-right corner */}
      <div className="absolute right-4 top-1/2 -translate-y-1/2">
        <ThemeToggle />
      </div>
    </header>
  );
}
