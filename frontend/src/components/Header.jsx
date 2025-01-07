import React from "react";
function Header() {
  return (
    <header className="bg-white">
      <div className="hidden lg:flex lg:gap-x-12">
        Breast Cancer Detection with AI
      </div>
      <nav>
        <ul>
          <li>
            <a href="/" className="text-sm/6 font-semibold text-gray-900">
              Home
            </a>
          </li>
          <li>
            <a href="/about" className="text-sm/6 font-semibold text-gray-900">
              About
            </a>
          </li>
          <li>
            <a
              href="/contact"
              className="text-sm/6 font-semibold text-gray-900"
            >
              Contact
            </a>
          </li>
        </ul>
      </nav>
    </header>
  );
}
export default Header;
