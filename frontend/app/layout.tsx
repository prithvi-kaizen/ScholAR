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
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
