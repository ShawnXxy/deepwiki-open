'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

/**
 * Wiki generation progress state
 * Tracks the state of wiki generation that can be minimized/restored
 */
interface WikiGenerationProgress {
  owner: string;
  repo: string;
  repoType: string;
  repoUrl?: string;
  totalPages: number;
  completedPages: number;
  isGenerating: boolean;
  isPaused?: boolean;
  currentPageId?: string;
  wikiUrl: string; // URL to navigate back to the wiki page
  language: string;
  branch?: string | null;
  comprehensive?: boolean; // Track comprehensive vs concise mode
}

interface WikiGenerationContextType {
  progress: WikiGenerationProgress | null;
  setProgress: (progress: WikiGenerationProgress | null) => void;
  isMinimized: boolean;
  minimize: () => void;
  restore: () => void;
  clearProgress: () => void;
  showCompletionNotification: boolean;
  setShowCompletionNotification: (show: boolean) => void;
  isGeneratingInBackground: boolean;
  setIsGeneratingInBackground: (value: boolean) => void;
}

const WikiGenerationContext = createContext<WikiGenerationContextType | undefined>(undefined);

export function WikiGenerationProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<WikiGenerationProgress | null>(null);
  const [isMinimized, setIsMinimized] = useState(false);
  const [showCompletionNotification, setShowCompletionNotification] = useState(false);
  const [isGeneratingInBackground, setIsGeneratingInBackground] = useState(false);

  const minimize = useCallback(() => {
    setIsMinimized(true);
  }, []);

  const restore = useCallback(() => {
    setIsMinimized(false);
  }, []);

  const clearProgress = useCallback(() => {
    setProgress(null);
    setIsMinimized(false);
    setShowCompletionNotification(false);
  }, []);

  return (
    <WikiGenerationContext.Provider
      value={{
        progress,
        setProgress,
        isMinimized,
        minimize,
        restore,
        clearProgress,
        showCompletionNotification,
        setShowCompletionNotification,
        isGeneratingInBackground,
        setIsGeneratingInBackground,
      }}
    >
      {children}
    </WikiGenerationContext.Provider>
  );
}

export function useWikiGeneration() {
  const context = useContext(WikiGenerationContext);
  if (context === undefined) {
    throw new Error('useWikiGeneration must be used within a WikiGenerationProvider');
  }
  return context;
}
