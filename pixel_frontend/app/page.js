/*
  This is the only page of the site (a single-page app).
  It stacks: Header → Upload/Analyser → About → FAQs → Footer
*/
import Header from "@/components/Header";
import UploadAnalyser from "@/components/UploadAnalyser";
import AboutUs from "@/components/AboutUs";
import Faqs from "@/components/Faqs";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="flex min-h-full flex-col">
      <Header />
      <main className="flex-1 pb-8">
        <UploadAnalyser />
        <AboutUs />
        <Faqs />
      </main>
      <Footer />
    </div>
  );
}
