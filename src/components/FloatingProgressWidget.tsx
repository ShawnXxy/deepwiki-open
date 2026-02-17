'use client';

import React, { useState } from 'react';
import { FaChevronDown, FaChevronUp, FaPlay, FaSpinner } from 'react-icons/fa';
import { useRouter } from 'next/navigation';
import { useWikiGeneration } from '@/contexts/WikiGenerationContext';
import { useLanguage } from '@/contexts/LanguageContext';

/**
 * FloatingProgressWidget
 * Displays minimized wiki generation progress in bottom-left corner
 * Allows user to expand/collapse details and restore to full view
 */
export default function FloatingProgressWidget() {
  const router = useRouter();
  const { progress, restore, isMinimized } = useWikiGeneration();
  const { messages } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(true);

  // Only show when progress exists AND user has minimized
  if (!progress || !isMinimized) {
    return null;
  }

  const progressPercentage = progress.totalPages > 0 
    ? Math.round((progress.completedPages / progress.totalPages) * 100) 
    : 0;

  const handleRestore = () => {
    restore();
    // Navigate to wiki page URL with all necessary parameters
    // Include: comprehensive mode, repo type, branch, and repoUrl for non-GitHub repos
    const params = new URLSearchParams();
    
    // Always include comprehensive param (page defaults to true if not 'false')
    params.set('comprehensive', progress.comprehensive === false ? 'false' : 'true');
    
    // Include type for non-github repos
    if (progress.repoType && progress.repoType !== 'github') {
      params.set('type', progress.repoType);
    }
    
    // Include branch if specified
    if (progress.branch) {
      params.set('branch', progress.branch);
    }
    
    // Include repo_url if available (needed for azuredevops and other non-standard repos)
    if (progress.repoUrl) {
      params.set('repo_url', encodeURIComponent(progress.repoUrl));
    }
    
    const resumeUrl = `/${progress.owner}/${progress.repo}?${params.toString()}`;
    router.push(resumeUrl);
  };

  return (
    <div className="fixed bottom-6 left-6 z-50 w-80 bg-[var(--card-bg)] rounded-lg shadow-2xl border-2 border-[var(--accent-primary)]/30 overflow-hidden">
      {/* Header */}
      <div className="bg-[var(--accent-primary)]/10 px-4 py-3 flex items-center justify-between border-b border-[var(--border-color)]">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {progress.isGenerating && (
            <FaSpinner className="text-[var(--accent-primary)] animate-spin flex-shrink-0" />
          )}
          <h3 className="text-sm font-semibold text-[var(--foreground)] truncate">
            {progress.owner}/{progress.repo}
          </h3>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-[var(--background)]/50 rounded transition-colors"
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <FaChevronDown className="text-[var(--muted)] text-sm" />
            ) : (
              <FaChevronUp className="text-[var(--muted)] text-sm" />
            )}
          </button>
        </div>
      </div>

      {/* Content - shown when expanded */}
      {isExpanded && (
        <div className="p-4 space-y-3">
          {/* Status */}
          <div className="text-xs text-[var(--muted)]">
            {progress.isPaused ? (
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
                {messages.wikiProgress?.paused || 'Generation paused'}
              </span>
            ) : progress.isGenerating ? (
              messages.wikiProgress?.generating || 'Generating wiki...'
            ) : (
              messages.wikiProgress?.completed || 'Generation completed'
            )}
          </div>

          {/* Progress Bar */}
          <div>
            <div className="bg-[var(--background)]/50 rounded-full h-2 overflow-hidden border border-[var(--border-color)]">
              <div
                className="bg-[var(--accent-primary)] h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(5, progressPercentage)}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between text-xs text-[var(--muted)]">
              <span>
                {messages.wikiProgress?.pagesCompleted
                  ?.replace('{completed}', progress.completedPages.toString())
                  ?.replace('{total}', progress.totalPages.toString()) ||
                  `${progress.completedPages} / ${progress.totalPages} pages`}
              </span>
              <span>{progressPercentage}%</span>
            </div>
          </div>

          {/* Actions - only show when paused */}
          {progress.isPaused && (
            <>
              <div className="flex gap-2 pt-2">
                <button
                  onClick={handleRestore}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-[var(--accent-primary)] text-white rounded-md hover:bg-[var(--highlight)] transition-colors font-medium text-sm"
                >
                  <FaPlay className="text-xs" />
                  {messages.wikiProgress?.resume || 'Resume Generation'}
                </button>
              </div>
              
              {/* Paused notice */}
              <div className="mt-2 text-xs text-[var(--muted)] text-center">
                ⏸️ {messages.wikiProgress?.pausedNotice || 'Generation paused - click Resume to continue'}
              </div>
            </>
          )}

          {/* Branch & Language info */}
          {(progress.branch || progress.language) && (
            <div className="flex flex-wrap gap-2 text-xs pt-2 border-t border-[var(--border-color)]">
              {progress.branch && (
                <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                  {progress.branch}
                </span>
              )}
              {progress.language && (
                <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/30">
                  {progress.language}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
