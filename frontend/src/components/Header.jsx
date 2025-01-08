import React from "react";
function Header() {
  return (
    <header className="bg-indigo-600">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" aria-label="Top">
        <div className="w-full py-6 flex items-center justify-between border-b border-indigo-500 lg:border-none">
          <div className="flex items-center">
            <a href="#">
              <span className="ml-3 text-white text-xl font-semibold">
                LumenAI
              </span>
            </a>
            <div className="hidden ml-10 space-x-8 lg:block">
              <a
                href="/"
                className="text-base font-medium text-white hover:text-indigo-50"
              >
                {" "}
                Home{" "}
              </a>

              <a
                href="/about"
                className="text-base font-medium text-white hover:text-indigo-50"
              >
                {" "}
                About{" "}
              </a>

              <a
                href="/contact"
                className="text-base font-medium text-white hover:text-indigo-50"
              >
                {" "}
                Contact{" "}
              </a>
            </div>
          </div>
          <div className="ml-10 space-x-4">
            <a
              href="#"
              className="inline-block bg-indigo-500 py-2 px-4 border border-transparent rounded-md text-base font-medium text-white hover:bg-opacity-75"
            >
              Sign in
            </a>
            <a
              href="#"
              className="inline-block bg-white py-2 px-4 border border-transparent rounded-md text-base font-medium text-indigo-600 hover:bg-indigo-50"
            >
              Sign up
            </a>
          </div>
        </div>
        <div className="py-4 flex flex-wrap justify-center space-x-6 lg:hidden">
          <a
            href="#"
            className="text-base font-medium text-white hover:text-indigo-50"
          >
            {" "}
            Home{" "}
          </a>

          <a
            href="#"
            className="text-base font-medium text-white hover:text-indigo-50"
          >
            {" "}
            About{" "}
          </a>

          <a
            href="#"
            className="text-base font-medium text-white hover:text-indigo-50"
          >
            {" "}
            Contact{" "}
          </a>
        </div>
      </nav>
    </header>
  );
}
export default Header;
