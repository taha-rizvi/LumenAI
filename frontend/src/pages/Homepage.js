import React from "react";
import Header from "../components/Header.jsx";
import HeroSection from "../components/hero_section.jsx";
import Features from "../components/features.jsx";
import HowItWorks from "../components/HowItWorks.jsx";
import FileUpload from "../components/file_upload_section.jsx";
import Footer from "../components/Footer.jsx";
function Homepage() {
  return (
    <div>
      <Header />
      <HeroSection />
      <Features />
      <HowItWorks />
      <FileUpload />
      <Footer />
    </div>
  );
}
export default Homepage;
