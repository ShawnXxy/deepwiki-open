import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "next-themes";
import { LanguageProvider } from "@/contexts/LanguageContext";

// Using system fonts for better Docker build compatibility (no network fetch needed)
// System font stacks provide good cross-platform support including CJK characters

export const metadata: Metadata = {
  title: "Orcas CodeWiki",
  description: "AI-powered documentation for your code repositories",
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
          <LanguageProvider>
            {children}
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
