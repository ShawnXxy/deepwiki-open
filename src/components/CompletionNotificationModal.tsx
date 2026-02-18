'use client';

import React from 'react';
import { FaCheck, FaTimes } from 'react-icons/fa';
import { useRouter } from 'next/navigation';
import { useWikiGeneration } from '@/contexts/WikiGenerationContext';
import { useLanguage } from '@/contexts/LanguageContext';

/**
 * CompletionNotificationModal
 * Shows a modal notification when wiki generation is completed
 * Offers options to view the wiki or dismiss
 */
export default function CompletionNotificationModal() {
  const router = useRouter();
  const { progress, showCompletionNotification, setShowCompletionNotification, clearProgress } = useWikiGeneration();
  const { messages } = useLanguage();

  if (!showCompletionNotification || !progress) {
    return null;
  }

  const handleViewWiki = () => {
    setShowCompletionNotification(false);
    clearProgress();
    router.push(progress.wikiUrl);
  };

  const handleDismiss = () => {
    setShowCompletionNotification(false);
    clearProgress();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-[var(--card-bg)] rounded-lg shadow-2xl border border-[var(--border-color)] max-w-md w-full mx-4 overflow-hidden">
        {/* Success Icon Header */}
        <div className="bg-[var(--accent-primary)]/10 px-6 py-8 text-center border-b border-[var(--border-color)]">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-500/20 border-2 border-green-500 mb-4">
            <FaCheck className="text-3xl text-green-600 dark:text-green-400" />
          </div>
          <h2 className="text-xl font-semibold text-[var(--foreground)]">
            {messages.wikiProgress?.completionTitle || 'Wiki Generation Complete!'}
          </h2>
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          <p className="text-[var(--foreground)] text-center mb-2">
            {messages.wikiProgress?.completionMessage
              ?.replace('{repo}', `${progress.owner}/${progress.repo}`) ||
              `Wiki for ${progress.owner}/${progress.repo} has been successfully generated.`}
          </p>
          <p className="text-sm text-[var(--muted)] text-center">
            {messages.wikiProgress?.completionSubtext ||
              'Would you like to view the wiki now?'}
          </p>
        </div>

        {/* Actions */}
        <div className="px-6 pb-6 flex gap-3">
          <button
            onClick={handleDismiss}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 transition-colors border border-[var(--border-color)]"
          >
            <FaTimes className="text-sm" />
            {messages.common?.close || 'Close'}
          </button>
          <button
            onClick={handleViewWiki}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-[var(--accent-primary)] text-white rounded-md hover:bg-[var(--accent-primary)]/90 transition-colors font-medium"
          >
            {messages.wikiProgress?.viewWiki || 'View Wiki'}
          </button>
        </div>
      </div>
    </div>
  );
}
