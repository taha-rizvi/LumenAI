import React from "react";
function HowItWorks() {
  return (
    <div className="relative bg-white py-16 sm:py-24 lg:py-32">
      <h2 className="mt-2 text-3xl font-extrabold tracking-tight text-gray-900 sm:text-4xl">
        How it Works?
      </h2>
      <ol>
        <li className="mx-auto mt-5 max-w-prose text-xl text-gray-500">
          Step 1: Upload the fna image of your report.
        </li>
        <li className="mx-auto mt-5 max-w-prose text-xl text-gray-500">
          Step 2: click the submit button and wait for a few seconds.
        </li>
        <li className="mx-auto mt-5 max-w-prose text-xl text-gray-500">
          Our application will process your input and present the result within
          seconds.
        </li>
      </ol>
    </div>
  );
}
export default HowItWorks;
