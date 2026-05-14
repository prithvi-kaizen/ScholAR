import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ScholAR",
  description: "Local LLM research paper assistant"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <script
          dangerouslySetInnerHTML={{
            __html: `
try {
  if (window.localStorage.getItem("scholar-theme") === "light") {
    document.documentElement.classList.add("theme-light");
  }
} catch (_) {}
            `.trim()
          }}
        />
        {children}
      </body>
    </html>
  );
}
