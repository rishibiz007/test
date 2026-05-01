import type { Metadata } from "next";
import "./globals.css";
import AmplitudeInit from "@/components/AmplitudeInit";

export const metadata: Metadata = {
  title: "Spark — Networking copilot",
  description:
    "Spark turns scattered public info about a person into 3-5 personal, timely, non-creepy talking points in under 30 seconds.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,500;0,600;1,400;1,500&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <AmplitudeInit />
        {children}
      </body>
    </html>
  );
}
