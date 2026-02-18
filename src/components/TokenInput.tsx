'use client';

import React, { useEffect, useMemo } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

type Platform = 'github' | 'gitlab' | 'bitbucket' | 'azuredevops';

interface TokenInputProps {
  repositoryUrl?: string;
  selectedPlatform: Platform;
  setSelectedPlatform: (value: Platform) => void;
  accessToken: string;
  setAccessToken: (value: string) => void;
  showTokenSection?: boolean;
  onToggleTokenSection?: () => void;
  allowPlatformChange?: boolean;
}

/**
 * Auto-detect platform from repository URL
 */
function detectPlatform(url: string): Platform | null {
  if (!url) return null;
  const lowerUrl = url.toLowerCase();
  
  if (lowerUrl.includes('dev.azure.com') || lowerUrl.includes('visualstudio.com')) {
    return 'azuredevops';
  }
  if (lowerUrl.includes('github.com')) {
    return 'github';
  }
  if (lowerUrl.includes('gitlab.com') || lowerUrl.includes('gitlab.')) {
    return 'gitlab';
  }
  if (lowerUrl.includes('bitbucket.org') || lowerUrl.includes('bitbucket.')) {
    return 'bitbucket';
  }
  return null;
}

export default function TokenInput({
  repositoryUrl = '',
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  showTokenSection = true,
  onToggleTokenSection,
}: TokenInputProps) {
  const { messages: t } = useLanguage();

  // Auto-detect platform from URL
  const detectedPlatform = useMemo(() => detectPlatform(repositoryUrl), [repositoryUrl]);
  const isAzureDevOps = detectedPlatform === 'azuredevops';
  
  // Update platform when URL changes
  useEffect(() => {
    if (detectedPlatform) {
      setSelectedPlatform(detectedPlatform);
    }
  }, [detectedPlatform, setSelectedPlatform]);

  const platformName = selectedPlatform === 'azuredevops' ? 'Azure DevOps' : 
                     selectedPlatform.charAt(0).toUpperCase() + selectedPlatform.slice(1);

  // For Azure DevOps, always show PAT (private repos need authentication)
  const shouldShowContent = isAzureDevOps || showTokenSection;

  return (
    <div className="mb-4">
      {/* Only show toggle button for non-ADO platforms */}
      {onToggleTokenSection && !isAzureDevOps && (
        <button
          type="button"
          onClick={onToggleTokenSection}
          className="text-sm text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5 mb-2"
        >
          {showTokenSection ? t.form?.hideTokens || 'Hide Access Tokens' : t.form?.addTokens || 'Add Access Tokens for Private Repositories'}
        </button>
      )}

      {shouldShowContent && (
        <div className="mt-2 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
          {/* Platform auto-detected indicator */}
          {detectedPlatform && (
            <div className="mb-3 flex items-center text-xs text-[var(--muted)]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Platform detected: <span className="font-medium ml-1">{platformName}</span>
            </div>
          )}

          <div>
            <label htmlFor="access-token" className="block text-xs font-medium text-[var(--foreground)] mb-2">
              {(t.form?.personalAccessToken || '{platform} Personal Access Token').replace('{platform}', platformName)}
            </label>
            <input
              id="access-token"
              type="password"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              placeholder={(t.form?.tokenPlaceholder || 'Enter your {platform} access token').replace('{platform}', platformName)}
              className="input-azure block w-full px-3 py-2 rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)] text-sm"
            />
            <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {t.form?.tokenSecurityNote || 'Required for private repositories. Your token is stored locally and never sent to our servers.'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 