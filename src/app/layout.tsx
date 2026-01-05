import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "next-themes";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { WikiGenerationProvider } from "@/contexts/WikiGenerationContext";
import FloatingProgressWidget from "@/components/FloatingProgressWidget";
import CompletionNotificationModal from "@/components/CompletionNotificationModal";
import BackgroundGenerationManager from "@/components/BackgroundGenerationManager";

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
            <WikiGenerationProvider>
              <BackgroundGenerationManager />
              {children}
              <FloatingProgressWidget />
              <CompletionNotificationModal />
            </WikiGenerationProvider>
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
