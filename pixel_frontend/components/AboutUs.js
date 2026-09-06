export default function AboutUs() {
  return (
    <section id="about" className="mx-auto w-full max-w-3xl px-4 py-16">
      <h2 className="text-xl font-semibold text-[var(--primary)] dark:text-blue-200">About Us</h2>
      <div className="mt-4 space-y-4 text-sm leading-7 text-[var(--muted)] sm:text-base">
        <p>
          PIXEL is a Project-Based Learning (PBL) project developed by students of the{" "}
          <strong className="font-medium text-[var(--foreground)]">
            B.Tech Computer Science and Engineering program at Sikkim Manipal Institute of Technology (SMIT)
          </strong>
          .
        </p>
        <p>
          The project is developed by the <strong className="font-medium text-[var(--foreground)]">2024–2028 batch</strong> as
          part of our academic learning and practical implementation. PIXEL focuses on creating a simple and
          intuitive platform for uploading and processing{" "}
          <strong className="font-medium text-[var(--foreground)]">images and videos</strong> through a clean, minimal, and
          user-friendly interface.
        </p>
        <p>
          This project represents our effort to apply the concepts and technologies learned throughout our B.Tech
          program to build a functional real-world application.
        </p>
      </div>
    </section>
  );
}
