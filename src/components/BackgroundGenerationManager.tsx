'use client';

import { useEffect, useRef } from 'react';
import { useWikiGeneration } from '@/contexts/WikiGenerationContext';

/**
 * BackgroundGenerationManager
 * Persistent component that manages wiki generation state across navigation
 * Polls server cache to keep progress updated even when page component unmounts
 * Lives at layout level, so it never unmounts
 */
export default function BackgroundGenerationManager() {
  const { progress, setProgress } = useWikiGeneration();
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Only poll if generation is paused (user navigated away from wiki page)
    // When user is on the wiki page, the page component handles its own state
    if (!progress || !progress.isGenerating || !progress.isPaused) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    console.log('[BackgroundManager] Starting polling for', progress.owner, progress.repo, '(paused state)');

    const pollProgress = async () => {
      try {
        const params = new URLSearchParams({
          owner: progress.owner,
          repo: progress.repo,
          repo_type: progress.repoType,
          language: progress.language,
          comprehensive: (progress.comprehensive ?? true).toString(),
        });

        if (progress.branch) {
          params.append('branch', progress.branch);
        }

        const response = await fetch(`/api/wiki_cache?${params.toString()}`);
        
        if (response.ok) {
          const cachedData = await response.json();
          
          if (cachedData?.wiki_structure && cachedData?.generated_pages) {
            const totalPages = cachedData.wiki_structure.pages?.length || 0;
            const generatedPagesArray = Object.values(cachedData.generated_pages) as Array<{ content?: string }>;
            const pagesWithContent = generatedPagesArray.filter(
              (p) => p.content && p.content !== 'Loading...' && !p.content.startsWith('Error')
            ).length;
            
            const isComplete = !cachedData.is_partial && pagesWithContent >= totalPages && totalPages > 0;
            
            // Update progress if changed
            if (progress.totalPages !== totalPages || 
                progress.completedPages !== pagesWithContent || 
                progress.isGenerating === isComplete) {
              
              console.log('[BackgroundManager] Progress updated:', pagesWithContent, '/', totalPages, isComplete ? '(complete)' : '(generating)');
              
              setProgress({
                ...progress,
                totalPages,
                completedPages: pagesWithContent,
                isGenerating: !isComplete,
              });
            }
          }
        }
      } catch (error) {
        console.error('[BackgroundManager] Error polling progress:', error);
      }
    };

    // Poll immediately, then every 3 seconds
    pollProgress();
    pollIntervalRef.current = setInterval(pollProgress, 3000);

    return () => {
      if (pollIntervalRef.current) {
        console.log('[BackgroundManager] Stopping polling');
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [progress, setProgress]);

  // This component renders nothing - it's just for side effects
  return null;
}
