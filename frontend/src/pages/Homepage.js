import React from "react";
import Header from "../components/Header";
import HeroSection from "../components/hero_section";
import Features from "../components/features";
import HowItWorks from "../components/HowItWorks";
import FileUpload from "../components/file_upload_section";
import Footer from "../components/Footer";
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
