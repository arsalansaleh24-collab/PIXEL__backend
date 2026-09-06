const FAQS = [
  {
    q: "What is PIXEL?",
    a: "PIXEL is a web-based file-processing platform that allows users to upload images and videos through a simple drag-and-drop interface.",
  },
  {
    q: "What type of files can I upload?",
    a: "PIXEL supports image and video files. Users can either drag and drop their files into the upload area or select a file from their device.",
  },
  {
    q: "Who developed PIXEL?",
    a: "PIXEL was developed as a Project-Based Learning (PBL) project by B.Tech Computer Science and Engineering students of Sikkim Manipal Institute of Technology, 2024–2028 batch.",
  },
];

export default function Faqs() {
  return (
    <section id="faqs" className="mx-auto w-full max-w-3xl px-4 pb-12">
      <h2 className="text-xl font-semibold text-[var(--primary)] dark:text-blue-200">
        Frequently Asked Questions
      </h2>
      <div className="mt-6 space-y-6">
        {FAQS.map((item) => (
          <div key={item.q}>
            <h3 className="text-base font-medium">{item.q}</h3>
            <p className="mt-2 text-sm leading-7 text-[var(--muted)] sm:text-base">{item.a}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
