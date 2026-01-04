'use client';

import React, { useState, useEffect } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

// Define the interfaces for model configuration
interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface ModelConfig {
  providers: Provider[];
  defaultProvider: string;
}

interface FiltersConfig {
  excluded_dirs: string[];
  excluded_files: string[];
}

interface ModelSelectorProps {
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;

  // File filter configuration
  showFileFilters?: boolean;
  excludedDirs?: string;
  setExcludedDirs?: (value: string) => void;
  excludedFiles?: string;
  setExcludedFiles?: (value: string) => void;
  includedDirs?: string;
  setIncludedDirs?: (value: string) => void;
  includedFiles?: string;
  setIncludedFiles?: (value: string) => void;
}

export default function UserSelector({
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel: _isCustomModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  setIsCustomModel: _setIsCustomModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  customModel: _customModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  setCustomModel: _setCustomModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  showFileFilters = false,
  excludedDirs = '',
  setExcludedDirs,
  excludedFiles = '',
  setExcludedFiles,
  includedDirs = '',
  setIncludedDirs,
  includedFiles = '',
  setIncludedFiles
}: ModelSelectorProps) {
  // State to manage the visibility of the filter section
  const [isFilterSectionOpen, setIsFilterSectionOpen] = useState(false);
  // State to manage filter mode: 'exclude' or 'include'
  const [filterMode, setFilterMode] = useState<'exclude' | 'include'>('exclude');
  const { messages: t } = useLanguage();

  // State for model configurations from backend
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // State for file filters from backend
  const [filtersConfig, setFiltersConfig] = useState<FiltersConfig | null>(null);

  // State for viewing default values
  const [showDefaultDirs, setShowDefaultDirs] = useState(false);
  const [showDefaultFiles, setShowDefaultFiles] = useState(false);

  // Fetch file filters configuration from the backend
  useEffect(() => {
    const fetchFiltersConfig = async () => {
      try {
        const response = await fetch('/api/filters/config');
        if (response.ok) {
          const data = await response.json();
          setFiltersConfig(data);
        } else {
          console.warn('Failed to fetch file filters config, all files will be embedded');
        }
      } catch (err) {
        console.warn('Failed to fetch file filters configuration, all files will be embedded:', err);
      }
    };
    fetchFiltersConfig();
  }, []);

  // Fetch model configurations from the backend
  useEffect(() => {
    const fetchModelConfig = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const response = await fetch('/api/models/config');

        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();
        setModelConfig(data);

        // Get the provider to use (current or default)
        const activeProvider = provider || data.defaultProvider;
        const selectedProvider = data.providers.find(
          (p: Provider) => p.id === activeProvider
        );

        // Initialize provider if not set
        if (!provider && data.defaultProvider) {
          setProvider(data.defaultProvider);
        }

        // Validate and fix model - ensure it matches available models from API
        // This handles stale localStorage values (e.g., old "gpt-4.1" cache)
        if (selectedProvider && selectedProvider.models.length > 0) {
          const validModelIds = selectedProvider.models.map((m: Model) => m.id);
          if (!model || !validModelIds.includes(model)) {
            // Model is empty or invalid - set to first available model
            setModel(selectedProvider.models[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
        setError('Failed to load model configurations. Using default options.');
        // Set fallback Azure configuration (should match infra.json)
        setModelConfig({
          defaultProvider: 'azure',
          providers: [{
            id: 'azure',
            name: 'Azure OpenAI',
            supportsCustomModel: false,
            models: [
              { id: 'o4-mini', name: 'o4-mini' }
            ]
          }]
        });
        if (!provider) {
          setProvider('azure');
          setModel('o4-mini');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelConfig();
  }, [provider, model, setModel, setProvider]);

  // Default excluded directories (from backend, empty if not loaded)
  const defaultExcludedDirs = filtersConfig?.excluded_dirs?.join('\n') || '';

  // Default excluded files (from backend, empty if not loaded)
  const defaultExcludedFiles = filtersConfig?.excluded_files?.join('\n') || '';

  // Display loading state
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <div className="text-sm text-[var(--muted)]">Loading configurations...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="space-y-4">
        {error && (
          <div className="text-sm text-red-500 mb-2">{error}</div>
        )}

        {/* File Filters Section */}
        {showFileFilters && (
          <div>
            <button
              type="button"
              onClick={() => setIsFilterSectionOpen(!isFilterSectionOpen)}
              className="flex items-center text-sm text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
            >
              <span className="mr-1.5 text-xs">{isFilterSectionOpen ? '▼' : '►'}</span>
              {t.form?.advancedOptions || 'Advanced Options'}
            </button>

            {isFilterSectionOpen && (
              <div className="mt-3 p-3 border border-[var(--border-color)]/70 rounded-md bg-[var(--background)]/30">
                {/* Filter Mode Selection */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                    {t.form?.filterMode || 'Filter Mode'}
                  </label>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setFilterMode('exclude')}
                      className={`flex-1 px-3 py-2 rounded-md border text-sm transition-colors ${
                        filterMode === 'exclude'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                      }`}
                    >
                      {t.form?.excludeMode || 'Exclude Paths'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setFilterMode('include')}
                      className={`flex-1 px-3 py-2 rounded-md border text-sm transition-colors ${
                        filterMode === 'include'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                      }`}
                    >
                      {t.form?.includeMode || 'Include Only Paths'}
                    </button>
                  </div>
                  <p className="text-xs text-[var(--muted)] mt-1">
                    {filterMode === 'exclude'
                      ? (t.form?.excludeModeDescription || 'Specify paths to exclude from processing')
                      : (t.form?.includeModeDescription || 'Specify only the paths to include')
                    }
                  </p>
                </div>

                {/* Directories Section */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-[var(--muted)] mb-1.5">
                    {filterMode === 'exclude'
                      ? (t.form?.excludedDirs || 'Excluded Directories')
                      : (t.form?.includedDirs || 'Included Directories')
                    }
                  </label>
                  <textarea
                    value={filterMode === 'exclude' ? excludedDirs : includedDirs}
                    onChange={(e) => {
                      if (filterMode === 'exclude') {
                        setExcludedDirs?.(e.target.value);
                      } else {
                        setIncludedDirs?.(e.target.value);
                      }
                    }}
                    rows={4}
                    className="block w-full rounded-md border border-[var(--border-color)]/50 bg-[var(--input-bg)] text-[var(--foreground)] px-3 py-2 text-sm focus:border-[var(--accent-primary)] focus:ring-1 focus:ring-opacity-50 shadow-sm"
                    placeholder={filterMode === 'exclude'
                      ? (t.form?.enterExcludedDirs || 'Enter excluded directories, one per line...')
                      : (t.form?.enterIncludedDirs || 'Enter included directories, one per line...')
                    }
                  />
                  {filterMode === 'exclude' && (
                    <>
                      <div className="flex mt-1.5">
                        <button
                          type="button"
                          onClick={() => setShowDefaultDirs(!showDefaultDirs)}
                          className="text-xs text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
                        >
                          {showDefaultDirs ? (t.form?.hideDefault || 'Hide Default') : (t.form?.viewDefault || 'View Default')}
                        </button>
                      </div>
                      {showDefaultDirs && (
                        <div className="mt-2 p-2 rounded bg-[var(--background)]/50 text-xs">
                          <p className="mb-1 text-[var(--muted)]">{t.form?.defaultNote || 'These defaults are already applied.'}</p>
                          <pre className="whitespace-pre-wrap font-mono text-[var(--muted)] overflow-y-auto max-h-32">{defaultExcludedDirs}</pre>
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Files Section */}
                <div>
                  <label className="block text-sm font-medium text-[var(--muted)] mb-1.5">
                    {filterMode === 'exclude'
                      ? (t.form?.excludedFiles || 'Excluded Files')
                      : (t.form?.includedFiles || 'Included Files')
                    }
                  </label>
                  <textarea
                    value={filterMode === 'exclude' ? excludedFiles : includedFiles}
                    onChange={(e) => {
                      if (filterMode === 'exclude') {
                        setExcludedFiles?.(e.target.value);
                      } else {
                        setIncludedFiles?.(e.target.value);
                      }
                    }}
                    rows={4}
                    className="block w-full rounded-md border border-[var(--border-color)]/50 bg-[var(--input-bg)] text-[var(--foreground)] px-3 py-2 text-sm focus:border-[var(--accent-primary)] focus:ring-1 focus:ring-opacity-50 shadow-sm"
                    placeholder={filterMode === 'exclude'
                      ? (t.form?.enterExcludedFiles || 'Enter excluded files, one per line...')
                      : (t.form?.enterIncludedFiles || 'Enter included files, one per line...')
                    }
                  />
                  {filterMode === 'exclude' && (
                    <>
                      <div className="flex mt-1.5">
                        <button
                          type="button"
                          onClick={() => setShowDefaultFiles(!showDefaultFiles)}
                          className="text-xs text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
                        >
                          {showDefaultFiles ? (t.form?.hideDefault || 'Hide Default') : (t.form?.viewDefault || 'View Default')}
                        </button>
                      </div>
                      {showDefaultFiles && (
                        <div className="mt-2 p-2 rounded bg-[var(--background)]/50 text-xs">
                          <p className="mb-1 text-[var(--muted)]">{t.form?.defaultNote || 'These defaults are already applied.'}</p>
                          <pre className="whitespace-pre-wrap font-mono text-[var(--muted)] overflow-y-auto max-h-32">{defaultExcludedFiles}</pre>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
