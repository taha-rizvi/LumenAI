"use client";
import React from "react";
function HeroSection() {
  return (
    <section className="bg-white">
      <h1 className="text-balance text-5xl font-semibold tracking-tight text-gray-900 sm:text-7xl">
        Detect and Treat Breast Cancer early with our fast and accurate AI
        System.{" "}
      </h1>
      <p className="mt-8 text-pretty text-lg font-medium text-gray-500 sm:text-xl/8">
        You just need an FNA image of your report and that's it.
      </p>
      <div className="mt-10 flex items-center justify-center gap-x-6">
        <a
          href="#"
          className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
        >
          Get Started
        </a>
      </div>
    </section>
  );
}
export default HeroSection;
