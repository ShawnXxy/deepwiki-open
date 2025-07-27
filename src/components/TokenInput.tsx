'use client';

import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

interface TokenInputProps {
  selectedPlatform: 'github' | 'gitlab' | 'bitbucket' | 'azuredevops';
  setSelectedPlatform: (value: 'github' | 'gitlab' | 'bitbucket' | 'azuredevops') => void;
  accessToken: string;
  setAccessToken: (value: string) => void;
  showTokenSection?: boolean;
  onToggleTokenSection?: () => void;
  allowPlatformChange?: boolean;
}

export default function TokenInput({
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  showTokenSection = true,
  onToggleTokenSection,
  allowPlatformChange = true
}: TokenInputProps) {
  const { messages: t } = useLanguage();

  const platformName = selectedPlatform === 'azuredevops' ? 'Azure DevOps' : 
                     selectedPlatform.charAt(0).toUpperCase() + selectedPlatform.slice(1);

  return (
    <div className="mb-4">
      {onToggleTokenSection && (
        <button
          type="button"
          onClick={onToggleTokenSection}
          className="text-sm text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5 mb-2"
        >
          {showTokenSection ? t.form?.hideTokens || 'Hide Access Tokens' : t.form?.addTokens || 'Add Access Tokens for Private Repositories'}
        </button>
      )}

      {showTokenSection && (
        <div className="mt-2 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
          {allowPlatformChange && (
            <div className="mb-3">
              <label className="block text-xs font-medium text-[var(--foreground)] mb-2">
                {t.form?.selectPlatform || 'Select Platform'}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedPlatform('github')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'github'
                    ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                    : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                    }`}
                >
                  <span className="text-sm">GitHub</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedPlatform('gitlab')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'gitlab'
                    ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                    : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                    }`}
                >
                  <span className="text-sm">GitLab</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedPlatform('bitbucket')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'bitbucket'
                    ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                    : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                    }`}
                >
                  <span className="text-sm">Bitbucket</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedPlatform('azuredevops')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'azuredevops'
                    ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                    : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                    }`}
                >
                  <span className="text-sm">Azure DevOps</span>
                </button>
              </div>
            </div>
          )}

          <div>
            <label htmlFor="access-token" className="block text-xs font-medium text-[var(--foreground)] mb-2">
              {(t.form?.personalAccessToken || 'Personal Access Token').replace('{platform}', platformName)}
            </label>
            <input
              id="access-token"
              type="password"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              placeholder={(t.form?.tokenPlaceholder || 'Enter your access token').replace('{platform}', platformName)}
              className="input-japanese block w-full px-3 py-2 rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)] text-sm"
            />
            <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {selectedPlatform === 'azuredevops' 
                ? 'Personal Access Token (PAT) is required for Azure DevOps repositories. Your token is stored locally and never sent to our servers.'
                : t.form?.tokenSecurityNote || 'Your token is stored locally and never sent to our servers.'
              }
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 