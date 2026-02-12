import "./globals.css";
import { IBM_Plex_Sans, Sora } from "next/font/google";

const sora = Sora({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["600", "700", "800"]
});

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"]
});

export const metadata = {
  title: "Dataset Quality Analyzer",
  description: "Pre-training dataset quality analyzer for duplicates, labels, toxicity, domain drift, and leakage."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${sora.variable} ${plex.variable}`}>{children}</body>
    </html>
  );
}
