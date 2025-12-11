import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Property Stream - Investment Recommendation Engine",
  description: "AI-powered property investment recommendations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

