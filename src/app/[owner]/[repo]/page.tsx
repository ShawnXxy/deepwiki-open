/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import Ask from '@/components/Ask';
import Markdown from '@/components/Markdown';
import ModelSelectionModal from '@/components/ModelSelectionModal';
import ThemeToggle from '@/components/theme-toggle';
import WikiTreeView from '@/components/WikiTreeView';
import { useLanguage } from '@/contexts/LanguageContext';
import { useWikiGeneration } from '@/contexts/WikiGenerationContext';
import { RepoInfo } from '@/types/repoinfo';
import { processCitations, generateFileUrl } from '@/utils/citationProcessor';
import { detectCurrentBranch } from '@/utils/branchDetection';
import getRepoUrl from '@/utils/getRepoUrl';
import logger from '@/utils/logger';
import { extractUrlDomain, extractUrlPath } from '@/utils/urlDecoder';
import Link from 'next/link';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaBitbucket, FaBookOpen, FaCog, FaComments, FaDownload, FaExclamationTriangle, FaFileExport, FaFolder, FaGithub, FaGitlab, FaHome, FaSync, FaTimes, FaMinusSquare } from 'react-icons/fa';
// Define the WikiSection and WikiStructure types directly in this file
// since the imported types don't have the sections and rootSections properties
interface WikiSection {
  id: string;
  title: string;
  pages: string[];
  subsections?: WikiSection[] | string[];
}

interface WikiPage {
  id: string;
  title: string;
  content: string;
  filePaths: string[];
  importance: 'high' | 'medium' | 'low';
  relatedPages: string[];
  parentId?: string;
  isSection?: boolean;
  children?: string[];
}

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];
}

// Add CSS styles for wiki with Japanese aesthetic
const wikiStyles = `
  .prose code {
    @apply bg-[var(--background)]/70 px-1.5 py-0.5 rounded font-mono text-xs border border-[var(--border-color)];
  }

  .prose pre {
    @apply bg-[var(--background)]/80 text-[var(--foreground)] rounded-md p-4 overflow-x-auto border border-[var(--border-color)] shadow-sm;
  }

  .prose h1, .prose h2, .prose h3, .prose h4 {
    @apply font-semibold text-[var(--foreground)];
  }

  .prose p {
    @apply text-[var(--foreground)] leading-relaxed;
  }

  .prose a {
    @apply text-[var(--accent-primary)] hover:text-[var(--highlight)] transition-colors no-underline border-b border-[var(--border-color)] hover:border-[var(--accent-primary)];
  }

  .prose blockquote {
    @apply border-l-4 border-[var(--accent-primary)]/30 bg-[var(--background)]/30 pl-4 py-1 italic;
  }

  .prose ul, .prose ol {
    @apply text-[var(--foreground)];
  }

  .prose table {
    @apply border-collapse border border-[var(--border-color)];
  }

  .prose th {
    @apply bg-[var(--background)]/70 text-[var(--foreground)] p-2 border border-[var(--border-color)];
  }

  .prose td {
    @apply p-2 border border-[var(--border-color)];
  }
`;

// Helper function to generate cache key for localStorage
// Branch is included in the key to support different branches of the same repo
const getCacheKey = (owner: string, repo: string, repoType: string, language: string, isComprehensive: boolean = true, branch?: string | null): string => {
  // Use 'default' for null/undefined/empty branch to maintain backwards compatibility
  const branchSuffix = branch?.trim() || 'default';
  return `deepwiki_cache_${repoType}_${owner}_${repo}_${language}_${isComprehensive ? 'comprehensive' : 'concise'}_${branchSuffix}`;
};

// Helper function to add tokens and other parameters to request body
const addTokensToRequestBody = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  requestBody: Record<string, any>,
  token: string,
  repoType: string,
  provider: string = '',
  model: string = '',
  isCustomModel: boolean = false,
  customModel: string = '',
  language: string = 'en',
  branch?: string,
  excludedDirs?: string,
  excludedFiles?: string,
  includedDirs?: string,
  includedFiles?: string,
  forceReprocess?: boolean
): void => {
  if (token !== '') {
    requestBody.token = token;
  }

  // Add branch parameter if provided
  if (branch) {
    requestBody.branch = branch;
  }

  // Add provider-based model selection parameters
  requestBody.provider = provider;
  requestBody.model = model;
  if (isCustomModel && customModel) {
    requestBody.custom_model = customModel;
  }

  requestBody.language = language;

  // Add file filter parameters if provided
  if (excludedDirs) {
    requestBody.excluded_dirs = excludedDirs;
  }
  if (excludedFiles) {
    requestBody.excluded_files = excludedFiles;
  }
  if (includedDirs) {
    requestBody.included_dirs = includedDirs;
  }
  if (includedFiles) {
    requestBody.included_files = includedFiles;
  }

  // Add force_reprocess flag for migration from pkl to vector-based storage
  if (forceReprocess) {
    requestBody.force_reprocess = true;
  }
};

const createGithubHeaders = (githubToken: string): HeadersInit => {
  const headers: HeadersInit = {
    'Accept': 'application/vnd.github.v3+json'
  };

  if (githubToken) {
    headers['Authorization'] = `Bearer ${githubToken}`;
  }

  return headers;
};

const createGitlabHeaders = (gitlabToken: string): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (gitlabToken) {
    headers['PRIVATE-TOKEN'] = gitlabToken;
  }

  return headers;
};

const createBitbucketHeaders = (bitbucketToken: string): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (bitbucketToken) {
    headers['Authorization'] = `Bearer ${bitbucketToken}`;
  }

  return headers;
};


export default function RepoWikiPage() {
  // Get route parameters and search params
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();

  // Wiki generation context for minimized progress
  const { progress, setProgress, isMinimized, minimize: minimizeProgress, restore, setShowCompletionNotification, setIsGeneratingInBackground, isGeneratingInBackground } = useWikiGeneration();

  // Extract owner and repo from route params
  const owner = params.owner as string;
  const repo = params.repo as string;

  // SECURITY: Retrieve token from sessionStorage (not URL params)
  // This prevents token exposure in server logs and browser history
  const [token, setToken] = useState<string>('');
  const [tokenChecked, setTokenChecked] = useState<boolean>(false);
  // Use ref to track token synchronously (avoids React state update race conditions)
  const tokenRef = useRef<string>('');
  
  useEffect(() => {
    // Try sessionStorage first (secure), fallback to URL params (legacy)
    const tokenKey = `deepwiki_token_${owner}_${repo}`;
    const storedToken = sessionStorage.getItem(tokenKey);
    const urlToken = searchParams.get('token') || '';
    
    logger.info('Token loading started', {
      owner,
      repo,
      tokenKey,
      hasStoredToken: !!storedToken,
      storedTokenLength: storedToken?.length || 0,
      hasUrlToken: !!urlToken
    });
    
    if (storedToken) {
      setToken(storedToken);
      tokenRef.current = storedToken;
      logger.info('Token loaded from sessionStorage', { tokenLength: storedToken.length });
      // Clear token from URL if it exists (for security)
      if (urlToken) {
        const url = new URL(window.location.href);
        url.searchParams.delete('token');
        window.history.replaceState({}, '', url.toString());
      }
    } else if (urlToken) {
      // Legacy support: use URL token but move it to sessionStorage
      setToken(urlToken);
      tokenRef.current = urlToken;
      sessionStorage.setItem(tokenKey, urlToken);
      logger.info('Token loaded from URL and saved to sessionStorage', { tokenLength: urlToken.length });
      // Remove from URL
      const url = new URL(window.location.href);
      url.searchParams.delete('token');
      window.history.replaceState({}, '', url.toString());
    } else {
      logger.info('No token found in sessionStorage or URL');
    }
    // Mark token as checked (even if empty) to allow main effect to proceed
    setTokenChecked(true);
    logger.debug('Token check completed', { tokenChecked: true });
  }, [owner, repo, searchParams]);
  const localPath = searchParams.get('local_path') ? decodeURIComponent(searchParams.get('local_path') || '') : undefined;
  const repoUrl = searchParams.get('repo_url') ? decodeURIComponent(searchParams.get('repo_url') || '') : undefined;
  const providerParam = searchParams.get('provider') || '';
  const modelParam = searchParams.get('model') || '';
  const isCustomModelParam = searchParams.get('is_custom_model') === 'true';
  const customModelParam = searchParams.get('custom_model') || '';
  const language = searchParams.get('language') || 'en';
  const branch = searchParams.get('branch') || null;
  const repoHost = (() => {
    if (!repoUrl) return '';
    try {
      return new URL(repoUrl).hostname.toLowerCase();
    } catch (e) {
      console.warn(`Invalid repoUrl provided: ${repoUrl}`);
      return '';
    }
  })();
  const repoType = repoHost?.includes('bitbucket')
    ? 'bitbucket'
    : repoHost?.includes('gitlab')
      ? 'gitlab'
      : repoHost?.includes('github')
        ? 'github'
        : repoUrl?.includes('dev.azure.com') || repoUrl?.includes('visualstudio.com')
          ? 'azuredevops'
          : searchParams.get('type') || 'github';

  // Import language context for translations
  const { messages } = useLanguage();

  // Initialize repo info
  const repoInfo = useMemo<RepoInfo>(() => ({
    owner,
    repo,
    type: repoType,
    token: token || null,
    branch: branch,
    localPath: localPath || null,
    repoUrl: repoUrl || null
  }), [owner, repo, repoType, localPath, repoUrl, token, branch]);

  // State variables
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerationStarted, setIsGenerationStarted] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | undefined>(
    messages.loading?.initializing || 'Initializing wiki generation...'
  );
  const [error, setError] = useState<string | null>(null);
  const [partialCacheMessage, setPartialCacheMessage] = useState<string | null>(null); // Info banner for partial cache
  const [wikiStructure, setWikiStructure] = useState<WikiStructure | undefined>();
  const [currentPageId, setCurrentPageId] = useState<string | undefined>();
  const [generatedPages, setGeneratedPages] = useState<Record<string, WikiPage>>({});
  const [pagesInProgress, setPagesInProgress] = useState(new Set<string>());
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [originalMarkdown, setOriginalMarkdown] = useState<Record<string, string>>({});
  const [requestInProgress, setRequestInProgress] = useState(false);
  const [currentToken, setCurrentToken] = useState(token); // Track current effective token
  const [effectiveRepoInfo, setEffectiveRepoInfo] = useState(repoInfo); // Track effective repo info with cached data
  const [embeddingError, setEmbeddingError] = useState(false);

  // Sync currentToken when token state changes (e.g., loaded from sessionStorage)
  useEffect(() => {
    if (token && token !== currentToken) {
      setCurrentToken(token);
      tokenRef.current = token;
    }
  }, [token, currentToken]);

  // Model selection state variables
  const [selectedProviderState, setSelectedProviderState] = useState(providerParam);
  const [selectedModelState, setSelectedModelState] = useState(modelParam);
  const [isCustomSelectedModelState, setIsCustomSelectedModelState] = useState(isCustomModelParam);
  const [customSelectedModelState, setCustomSelectedModelState] = useState(customModelParam);
  const [showModelOptions, setShowModelOptions] = useState(false); // Controls whether to show model options
  const excludedDirs = searchParams.get('excluded_dirs') || '';
  const excludedFiles = searchParams.get('excluded_files') || '';
  const [modelExcludedDirs, setModelExcludedDirs] = useState(excludedDirs);
  const [modelExcludedFiles, setModelExcludedFiles] = useState(excludedFiles);
  const includedDirs = searchParams.get('included_dirs') || '';
  const includedFiles = searchParams.get('included_files') || '';
  const [modelIncludedDirs, setModelIncludedDirs] = useState(includedDirs);
  const [modelIncludedFiles, setModelIncludedFiles] = useState(includedFiles);


  // Wiki type state - default to comprehensive view
  const isComprehensiveParam = searchParams.get('comprehensive') !== 'false';
  const [isComprehensiveView, setIsComprehensiveView] = useState(isComprehensiveParam);
  // Using useRef for activeContentRequests to maintain a single instance across renders
  // This map tracks which pages are currently being processed to prevent duplicate requests
  // Note: In a multi-threaded environment, additional synchronization would be needed,
  // but in React's single-threaded model, this is safe as long as we set the flag before any async operations
  const activeContentRequests = useRef(new Map<string, boolean>()).current;
  const [structureRequestInProgress, setStructureRequestInProgress] = useState(false);
  // Create a flag to track if data was loaded from cache to prevent immediate re-save
  const cacheLoadedSuccessfully = useRef(false);
  
  // Track last checkpoint save time to avoid too-frequent saves
  const lastCheckpointTime = useRef<number>(0);
  const CHECKPOINT_INTERVAL_MS = 10000; // Save checkpoint at most every 10 seconds

  // Keepalive worker ref — prevents browser from freezing the tab
  // during long wiki generation (embedding + page generation).
  // Web Workers are not throttled by background-tab rules.
  const keepAliveWorkerRef = useRef<Worker | null>(null);
  
  // Track if we're resuming from a partial cache
  const [isResumingFromPartial, setIsResumingFromPartial] = useState(false);
  
  // Ref to capture latest progress value for use in useEffect without causing re-triggers
  const progressRef = useRef(progress);
  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  // Keepalive worker: prevents browser from freezing/suspending the tab
  // during long-running wiki generation (embedding can take 30+ minutes).
  // A Web Worker's setInterval is NOT throttled by background-tab rules;
  // its postMessage wakes up the main thread's event loop.
  useEffect(() => {
    if (!isLoading) {
      // Not loading — terminate any existing keepalive worker
      if (keepAliveWorkerRef.current) {
        keepAliveWorkerRef.current.terminate();
        keepAliveWorkerRef.current = null;
      }
      return;
    }
    // Start a keepalive worker when wiki generation is in progress
    try {
      const code = 'setInterval(function(){postMessage(0)},15000)';
      const blob = new Blob([code], { type: 'text/javascript' });
      const url = URL.createObjectURL(blob);
      const worker = new Worker(url);
      worker.onmessage = () => {}; // Receiving messages keeps main thread alive
      keepAliveWorkerRef.current = worker;
      return () => {
        worker.terminate();
        URL.revokeObjectURL(url);
        keepAliveWorkerRef.current = null;
      };
    } catch {
      // Web Workers may be unavailable (e.g., SSR, restrictive CSP)
      console.warn('Could not create keepalive worker');
    }
  }, [isLoading]);

  // Sync isComprehensiveView when URL params change (useState only uses initial value once)
  useEffect(() => {
    setIsComprehensiveView(isComprehensiveParam);
  }, [isComprehensiveParam]);

  // Create a flag to ensure the effect only runs once
  const effectRan = React.useRef(false);

  // Flag to trigger force reprocessing (migration from pkl to vector-based storage)
  // Set to true in confirmRefresh, used once in determineWikiStructure, then reset
  const forceReprocessRef = React.useRef(false);

  // State for chat panel visibility (collapsed/expanded)
  const [isChatPanelCollapsed, setIsChatPanelCollapsed] = useState(false);
  const askComponentRef = useRef<{ clearConversation: () => void } | null>(null);

  // Authentication state
  const [authRequired, setAuthRequired] = useState<boolean>(false);
  const [authCode, setAuthCode] = useState<string>('');
  const [isAuthLoading, setIsAuthLoading] = useState<boolean>(true);

  // Default branch state
  const [defaultBranch, setDefaultBranch] = useState<string>('main');

  // Show page when user navigates back (clear background flag)
  useEffect(() => {
    // When component mounts, clear background flag and paused state
    if (isGeneratingInBackground) {
      setIsGeneratingInBackground(false);
    }
    
    // If there's progress for this repo that's paused, clear the paused flag
    if (progress && 
        progress.owner === owner && 
        progress.repo === repo && 
        progress.isPaused) {
      console.log('[WikiPage] Clearing paused flag on mount');
      setProgress({
        ...progress,
        isPaused: false,
      });
    }
    
    // When component unmounts while generating, mark as paused
    return () => {
      // Use ref to get latest progress value without adding to dependencies
      const currentProgress = progressRef.current;
      if (currentProgress && 
          currentProgress.isGenerating && 
          !currentProgress.isPaused &&
          currentProgress.owner === owner && 
          currentProgress.repo === repo) {
        console.log('[WikiPage] Component unmounting during generation - marking as paused');
        setProgress({
          ...currentProgress,
          isPaused: true,
        });
      }
    };
  }, [isGeneratingInBackground, setIsGeneratingInBackground, setProgress, owner, repo]);

  // Memoize repo info to avoid triggering updates in callbacks

  // Add useEffect to handle scroll reset
  useEffect(() => {
    // Scroll to top when currentPageId changes
    const wikiContent = document.getElementById('wiki-content');
    if (wikiContent) {
      wikiContent.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [currentPageId]);

  // Track wiki generation progress and update global context
  // Start tracking as soon as generation begins (when fetchRepositoryStructure is called)
  // This tracks the entire process: fetching repo, embedding, determining structure, generating pages
  useEffect(() => {
    if (isLoading && isGenerationStarted) {
      // Use clean URL without query params for navigation
      const wikiUrl = `/${owner}/${repo}`;
      
      // Calculate progress based on wiki structure if available
      let completedPages = 0;
      let totalPages = 0;
      
      if (wikiStructure) {
        totalPages = wikiStructure.pages.length;
        // Count pages with actual content (not loading or error)
        completedPages = wikiStructure.pages.filter(page => {
          const pageContent = generatedPages[page.id]?.content;
          return pageContent && pageContent !== 'Loading...' && !pageContent.startsWith('Error');
        }).length;
      }
      
      // Only update if values actually changed to avoid infinite loop
      if (!progress || 
          progress.owner !== owner || 
          progress.repo !== repo ||
          progress.completedPages !== completedPages ||
          progress.totalPages !== totalPages ||
          !progress.isGenerating ||
          progress.isPaused) {
        // When resuming (progress exists for this repo), preserve the comprehensive value
        // Don't overwrite with isComprehensiveView which may not have synced yet
        const comprehensiveValue = (progress?.owner === owner && progress?.repo === repo) 
          ? progress.comprehensive 
          : isComprehensiveView;
        
        setProgress({
          owner,
          repo,
          repoType,
          repoUrl: effectiveRepoInfo.repoUrl ?? undefined,
          totalPages,
          completedPages,
          isGenerating: true,
          isPaused: false,
          currentPageId,
          wikiUrl,
          language,
          branch: effectiveRepoInfo.branch,
          comprehensive: comprehensiveValue,
        });
      }
    }
  }, [isLoading, isGenerationStarted, wikiStructure, generatedPages, currentPageId, owner, repo, repoType, language, effectiveRepoInfo.repoUrl, effectiveRepoInfo.branch, searchParams, setProgress, progress]);

  // Handle wiki generation completion
  useEffect(() => {
    if (!isLoading && wikiStructure && progress && progress.isGenerating && progress.owner === owner && progress.repo === repo) {
      // Verify all pages actually have content before marking as complete
      const allPagesComplete = wikiStructure.pages.every(page => {
        const pageContent = generatedPages[page.id]?.content;
        return pageContent && pageContent !== 'Loading...' && !pageContent.startsWith('Error');
      });
      
      if (allPagesComplete) {
        console.log('All pages complete - marking generation as finished');
        // Wiki generation completed - update final state
        setProgress({
          ...progress,
          completedPages: wikiStructure.pages.length,
          isGenerating: false,
        });
        
        // Clear generation started flag
        setIsGenerationStarted(false);
        
        // Show completion notification if minimized
        if (isMinimized) {
          setShowCompletionNotification(true);
        }
      }
    }
  }, [isLoading, wikiStructure, generatedPages, progress, owner, repo, isMinimized, setProgress, setShowCompletionNotification]);

  // Handle restore from minimized state - clear progress when viewing completed wiki
  useEffect(() => {
    if (!isMinimized && progress && progress.owner === owner && progress.repo === repo && !progress.isGenerating && !isLoading) {
      // Clear progress since we're viewing the completed wiki
      setProgress(null);
    }
  }, [isMinimized, progress, owner, repo, isLoading, setProgress]);

  // Fetch authentication status on component mount
  useEffect(() => {
    const fetchAuthStatus = async () => {
      try {
        setIsAuthLoading(true);
        const response = await fetch('/api/auth/status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAuthRequired(data.auth_required);
      } catch (err) {
        console.error("Failed to fetch auth status:", err);
        // Assuming auth is required if fetch fails to avoid blocking UI for safety
        setAuthRequired(true);
      } finally {
        setIsAuthLoading(false);
      }
    };

    fetchAuthStatus();
  }, []);

  // Save checkpoint (partial cache) to preserve progress during wiki generation
  // This allows resumption if the process is interrupted
  const saveCheckpoint = useCallback(async (
    structure: WikiStructure,
    pages: Record<string, WikiPage>,
    isPartial: boolean = true
  ) => {
    // Skip if this was loaded from cache (avoid overwriting complete cache with partial)
    if (cacheLoadedSuccessfully.current && !isResumingFromPartial) {
      return;
    }
    
    // Rate limit checkpoint saves
    const now = Date.now();
    if (isPartial && now - lastCheckpointTime.current < CHECKPOINT_INTERVAL_MS) {
      return;
    }
    lastCheckpointTime.current = now;
    
    // Only save if we have at least one page with content
    const pagesWithContent = Object.values(pages).filter(
      p => p.content && p.content !== 'Loading...' && !p.content.startsWith('Error')
    );
    if (pagesWithContent.length === 0) {
      return;
    }
    
    try {
      const structureToCache = {
        ...structure,
        sections: structure.sections || [],
        rootSections: structure.rootSections || []
      };
      
      const dataToCache = {
        repo: effectiveRepoInfo,
        language: language,
        comprehensive: isComprehensiveView,
        wiki_structure: structureToCache,
        generated_pages: pages,
        provider: selectedProviderState,
        model: selectedModelState,
        is_partial: isPartial
      };
      
      // Use fire-and-forget pattern with short timeout to not block UI
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      
      fetch(`/api/wiki_cache`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataToCache),
        signal: controller.signal
      }).then(response => {
        clearTimeout(timeout);
        if (response.ok) {
          console.log(`Checkpoint saved: ${pagesWithContent.length}/${structure.pages.length} pages (${isPartial ? 'partial' : 'complete'})`);
        } else {
          console.warn('Failed to save checkpoint:', response.status);
        }
      }).catch(err => {
        clearTimeout(timeout);
        if (err.name !== 'AbortError') {
          console.warn('Error saving checkpoint:', err);
        }
      });
    } catch (err) {
      console.warn('Error preparing checkpoint:', err);
    }
  }, [effectiveRepoInfo, language, isComprehensiveView, selectedProviderState, selectedModelState, isResumingFromPartial]);

  // Generate content for a wiki page
  // Returns { success: boolean, error?: string } to indicate completion status
  const generatePageContent = useCallback(async (page: WikiPage, owner: string, repo: string): Promise<{ success: boolean; error?: string }> => {
    // Use effectiveToken to handle race condition where currentToken may be stale
    const effectiveToken = token || currentToken;
    
    return new Promise<{ success: boolean; error?: string }>(async (resolve) => {
      try {
        // Skip if content already exists and is valid (not loading/error placeholder)
        const existingContent = generatedPages[page.id]?.content;
        if (existingContent && existingContent !== 'Loading...' && !existingContent.startsWith('Error')) {
          console.log(`Page ${page.id} (${page.title}) already has content, skipping`);
          resolve({ success: true });
          return;
        }

        // Skip if this page is already being processed
        // Use a synchronized pattern to avoid race conditions
        if (activeContentRequests.get(page.id)) {
          console.log(`Page ${page.id} (${page.title}) is already being processed, skipping duplicate call`);
          resolve({ success: true });
          return;
        }

        // Mark this page as being processed immediately to prevent race conditions
        // This ensures that if multiple calls happen nearly simultaneously, only one proceeds
        activeContentRequests.set(page.id, true);

        // Validate repo info
        if (!owner || !repo) {
          throw new Error('Invalid repository information. Owner and repo name are required.');
        }

        // Mark page as in progress
        setPagesInProgress(prev => new Set(prev).add(page.id));
        // Don't set loading message for individual pages during queue processing

        const filePaths = page.filePaths;

        // Store the initially generated content BEFORE rendering/potential modification
        setGeneratedPages(prev => ({
          ...prev,
          [page.id]: { ...page, content: 'Loading...' } // Placeholder
        }));
        setOriginalMarkdown(prev => ({ ...prev, [page.id]: '' })); // Clear previous original

        // Make API call to generate page content
        logger.info('Starting content generation', { page: page.title, pageId: page.id });

        // Get repository URL
        const repoUrl = getRepoUrl(effectiveRepoInfo);

        // Wiki page prompt is built server-side by the backend promptstore.
        // The frontend sends only page metadata — the backend injects
        // commit-pinned URLs, page catalog, file summaries, and RAG context.
        // The message content is just the page title (used as RAG query).

        // Prepare request body — backend builds the full prompt
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const requestBody: Record<string, any> = {
          repo_url: repoUrl,
          type: effectiveRepoInfo.type,
          messages: [{
            role: 'user',
            content: page.title
          }],
          // Enable file-path-aware retrieval + backend prompt building
          wiki_page_request: true,
          page_id: page.id,
          page_title: page.title,
          page_file_paths: page.filePaths,
          page_related_pages: page.relatedPages || [],
        };

        // Add tokens if available - use effectiveToken to handle race condition
        addTokensToRequestBody(requestBody, effectiveToken, effectiveRepoInfo.type, selectedProviderState, selectedModelState, isCustomSelectedModelState, customSelectedModelState, language, effectiveRepoInfo.branch || undefined, modelExcludedDirs, modelExcludedFiles, modelIncludedDirs, modelIncludedFiles);

        // Use WebSocket for communication
        let content = '';

        try {
          // Create WebSocket URL with proper network detection
          const { getWebSocketUrl, getTimeoutConfig, shouldUseWebSocket } = await import('@/utils/networkConfig');
          
          // Only attempt WebSocket in localhost environments where port 8001 is accessible
          // In cloud deployments (Azure, etc.), skip directly to HTTP proxy
          if (!shouldUseWebSocket()) {
            console.log('Cloud environment detected, using HTTP proxy instead of WebSocket');
            throw new Error('Skip WebSocket in cloud environment');
          }
          
          const wsUrl = getWebSocketUrl();
          const timeouts = getTimeoutConfig();
          
          console.log(`Attempting WebSocket connection to: ${wsUrl}`);
          console.log(`Using timeout config:`, timeouts);

          // Create a new WebSocket connection
          const ws = new WebSocket(wsUrl);

          // Create a promise that resolves when the WebSocket connection is complete
          await new Promise<void>((resolve, reject) => {
            // eslint-disable-next-line prefer-const
            let connectionTimeout: NodeJS.Timeout;
            
            // Set up event handlers
            ws.onopen = () => {
              console.log(`WebSocket connection established for page: ${page.title}`);
              if (connectionTimeout) clearTimeout(connectionTimeout);
              try {
                // Send the request as JSON
                ws.send(JSON.stringify(requestBody));
                resolve();
              } catch (error) {
                console.error('Error sending WebSocket message:', error);
                reject(new Error('Failed to send WebSocket message'));
              }
            };

            ws.onerror = (error) => {
              logger.error('WebSocket error', { error: String(error), page: page.title });
              if (connectionTimeout) clearTimeout(connectionTimeout);
              reject(new Error('WebSocket connection failed'));
            };

            // If the connection doesn't open within the configured timeout, fall back to HTTP
            connectionTimeout = setTimeout(() => {
              logger.warn('WebSocket connection timeout, will fallback to HTTP', { page: page.title });
              ws.close();
              reject(new Error('WebSocket connection timeout'));
            }, timeouts.connectionTimeout);
          });

          // Create a promise that resolves when the WebSocket response is complete
          await new Promise<void>((resolve, reject) => {
            // Handle incoming messages
            ws.onmessage = (event) => {
              // Filter out keepalive messages (HTML comments used to keep connection alive during embedding)
              const data = event.data;
              if (data && !data.startsWith('<!-- keepalive')) {
                content += data;
              }
            };

            // Handle WebSocket close - check for abnormal closure
            ws.onclose = (event) => {
              logger.info('WebSocket connection closed', { page: page.title, code: event.code, reason: event.reason || 'none' });
              // Code 1000 = normal closure, 1006 = abnormal (no close frame)
              if (event.code !== 1000 && event.code !== 1005) {
                logger.warn('Abnormal WebSocket closure', { code: event.code, wasClean: event.wasClean, page: page.title });
              }
              resolve();
            };

            // Handle WebSocket errors
            ws.onerror = (error) => {
              logger.error('WebSocket error during message reception', { error: String(error), page: page.title });
              reject(new Error('WebSocket error during message reception'));
            };
          });
        } catch (wsError) {
          logger.warn('WebSocket error, falling back to HTTP', { error: String(wsError), page: page.title });

          // Fall back to HTTP if WebSocket fails
          const response = await fetch(`/api/chat/stream`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
          });

          if (!response.ok) {
            const errorText = await response.text().catch(() => 'No error details available');
            logger.error('API error in HTTP fallback', { status: response.status, error: errorText, page: page.title });
            throw new Error(`Error generating page content: ${response.status} - ${response.statusText}`);
          }

          // Process the response
          content = '';
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (!reader) {
            throw new Error('Failed to get response reader');
          }

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              content += decoder.decode(value, { stream: true });
            }
            // Ensure final decoding
            content += decoder.decode();
          } catch (readError) {
            logger.error('Error reading stream', { error: String(readError), page: page.title });
            throw new Error('Error processing response stream');
          }
        }

        // Clean up markdown delimiters
        content = content.replace(/^```markdown\s*/i, '').replace(/```\s*$/i, '');

        // Strip content filter warning — use the partial content that was received
        // and append a visible note so the user knows the page was truncated
        if (content.includes('[CONTENT_FILTER_WARNING]')) {
          logger.warn('Page content was partially truncated by content filter', { page: page.title, contentLength: content.length });
          content = content.replace(/\n*\[CONTENT_FILTER_WARNING][^]*/m, '');
          // Check if the remaining content is meaningful (more than just the <details> header)
          // Strip the <details>...</details> block to check actual content length
          const withoutDetails = content.replace(/<details>[\s\S]*?<\/details>/i, '').trim();
          if (withoutDetails.length < 100) {
            // Near-empty page — show a meaningful placeholder
            content = `# ${page.title}\n\n` +
              '> This page could not be generated due to Azure OpenAI content filtering. ' +
              'The source files for this topic may contain terms that triggered automated safety checks. ' +
              'This does not indicate any issue with the source code itself.\n';
          } else {
            content += '\n\n---\n\n> **Note:** This page was partially truncated by Azure OpenAI content filtering. The content above may be incomplete.\n';
          }
        }

        logger.info('Received content', { page: page.title, contentLength: content.length });
        
        // Check for error responses from the backend
        if (content.startsWith('Error:') || content.startsWith('Error preparing retriever')) {
          logger.error('Backend error in content', { page: page.id, content: content.substring(0, 200) });
          throw new Error(content);
        }
        
        // Check for empty or minimal content (might indicate connection issues)
        if (content.length < 50) {
          logger.warn('Suspiciously short content', { page: page.title, content: content.substring(0, 100), contentLength: content.length });
          if (content.length === 0) {
            throw new Error('No content received from backend - possible connection interruption');
          }
        }

        // Detect LLM asking a question instead of generating content.
        // This happens when RAG context is thin and the model requests
        // more information instead of writing the wiki page.
        const looksLikeQuestion = (
          !content.includes('# ') &&
          (content.startsWith('Could you') ||
           content.startsWith('Please provide') ||
           content.startsWith('I need') ||
           content.startsWith('Can you') ||
           content.includes('provide the list of relevant source files'))
        );
        if (looksLikeQuestion) {
          logger.warn('LLM returned a question instead of content, generating placeholder', { page: page.title, content: content.substring(0, 200) });
          content = `# ${page.title}\n\n` +
            '> This page could not be generated because the source files for this topic ' +
            'did not contain enough indexable content (e.g., binary files, images, or empty directories). ' +
            'Try refreshing the wiki or adding more relevant source files to this page\'s file list.\n';
        }

        // Store the FINAL generated content
        const updatedPage = { ...page, content };
        setGeneratedPages(prev => ({ ...prev, [page.id]: updatedPage }));
        // Store this as the original for potential mermaid retries
        setOriginalMarkdown(prev => ({ ...prev, [page.id]: content }));

        resolve({ success: true });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        // Log error with context to backend
        logger.error('Error generating content for page', { 
          pageId: page.id, 
          pageTitle: page.title, 
          error: errorMessage,
          repoUrl: effectiveRepoInfo?.repoUrl 
        });
        // Update page state to show error
        setGeneratedPages(prev => ({
          ...prev,
          [page.id]: { ...page, content: `Error generating content: ${errorMessage}` }
        }));
        setError(`Failed to generate content for ${page.title}.`);
        resolve({ success: false, error: errorMessage }); // Resolve even on error to unblock queue
      } finally {
        // Clear the processing flag for this page
        // This must happen in the finally block to ensure the flag is cleared
        // even if an error occurs during processing
        activeContentRequests.delete(page.id);

        // Mark page as done
        setPagesInProgress(prev => {
          const next = new Set(prev);
          next.delete(page.id);
          return next;
        });
        setLoadingMessage(undefined); // Clear specific loading message
      }
    });
  }, [generatedPages, token, currentToken, effectiveRepoInfo, selectedProviderState, selectedModelState, isCustomSelectedModelState, customSelectedModelState, modelExcludedDirs, modelExcludedFiles, modelIncludedDirs, modelIncludedFiles, language, activeContentRequests]);

  // Save checkpoint when pages are generated (to allow resumption on interruption)
  useEffect(() => {
    // Skip if no structure or if cache was just loaded
    if (!wikiStructure || cacheLoadedSuccessfully.current && !isResumingFromPartial) {
      return;
    }
    
    // Count pages with actual content (not loading or error)
    const pagesWithContent = Object.values(generatedPages).filter(
      p => p.content && p.content !== 'Loading...' && !p.content.startsWith('Error')
    );
    
    // Only save checkpoint if we have at least one page and not all pages are done
    const totalPages = wikiStructure.pages.length;
    if (pagesWithContent.length > 0 && pagesWithContent.length < totalPages) {
      saveCheckpoint(wikiStructure, generatedPages, true);
    }
  }, [generatedPages, wikiStructure, saveCheckpoint, isResumingFromPartial]);

  // Determine the wiki structure from repository data
  // detectedBranch is passed directly to avoid race condition with effectiveRepoInfo state update
  const determineWikiStructure = useCallback(async (fileTree: string, readme: string, owner: string, repo: string, detectedBranch?: string | null) => {
    if (!owner || !repo) {
      setError('Invalid repository information. Owner and repo name are required.');
      setIsLoading(false);
      setEmbeddingError(false); // Reset embedding error state
      return;
    }

    // Skip if structure request is already in progress
    if (structureRequestInProgress) {
      console.log('Wiki structure determination already in progress, skipping duplicate call');
      return;
    }

    // Use effectiveToken to handle race condition where currentToken may be stale
    const effectiveToken = token || currentToken;
    console.log('[determineWikiStructure] Using token:', {
      hasToken: !!token,
      hasCurrentToken: !!currentToken,
      hasEffectiveToken: !!effectiveToken,
      effectiveTokenLength: effectiveToken?.length || 0
    });

    try {
      setStructureRequestInProgress(true);
      setLoadingMessage(messages.loading?.determiningStructure || 'Determining wiki structure...');

      // Get repository URL
      const repoUrl = getRepoUrl(effectiveRepoInfo);

      // Truncate file tree if it's too large to prevent context overflow
      // o4-mini has 200k token limit, ~4 chars per token, so ~800k chars max
      // Leave room for prompt template, readme, and response (~200k chars for file tree)
      const MAX_FILE_TREE_CHARS = 200000;
      let truncatedFileTree = fileTree;
      const fileTreeTruncated = fileTree.length > MAX_FILE_TREE_CHARS;
      if (fileTreeTruncated) {
        // Keep the first portion of the file tree (most important structure)
        const lines = fileTree.split('\n');
        let charCount = 0;
        const keptLines: string[] = [];
        for (const line of lines) {
          if (charCount + line.length + 1 > MAX_FILE_TREE_CHARS) {
            break;
          }
          keptLines.push(line);
          charCount += line.length + 1;
        }
        truncatedFileTree = keptLines.join('\n');
        console.log(`[Wiki Structure] File tree truncated from ${fileTree.length} to ${truncatedFileTree.length} chars (${lines.length} to ${keptLines.length} files)`);
      }

      // Truncate readme if too large (leave ~50k chars for readme)
      const MAX_README_CHARS = 50000;
      let truncatedReadme = readme;
      if (readme.length > MAX_README_CHARS) {
        truncatedReadme = readme.substring(0, MAX_README_CHARS) + '\n\n[README truncated due to length...]';
        console.log(`[Wiki Structure] README truncated from ${readme.length} to ${truncatedReadme.length} chars`);
      }

      // Prepare request body with wiki_structure_request flag
      // Backend will use promptstore templates to build the actual prompt
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const requestBody: Record<string, any> = {
        repo_url: repoUrl,
        type: effectiveRepoInfo.type,
        // Signal to backend this is a wiki structure request
        wiki_structure_request: true,
        file_tree: truncatedFileTree,
        readme: truncatedReadme,
        comprehensive: isComprehensiveView,
        // Placeholder message - backend will replace with promptstore template
        messages: [{
          role: 'user',
          content: 'Generate wiki structure'  // Backend replaces this
        }]
      };

      // Add tokens if available - use effectiveToken to handle race condition
      // Use detectedBranch (passed directly) or effectiveRepoInfo.branch as fallback
      // This fixes the race condition where effectiveRepoInfo.branch state hasn't updated yet
      const branchToUse = detectedBranch || effectiveRepoInfo.branch || undefined;
      
      // Check if force_reprocess is requested (from Refresh Wiki action)
      // This triggers migration from pkl to vector-based storage
      const shouldForceReprocess = forceReprocessRef.current;
      if (shouldForceReprocess) {
        console.log('[determineWikiStructure] force_reprocess=true - will regenerate embeddings');
        forceReprocessRef.current = false; // Reset after use
      }
      
      addTokensToRequestBody(requestBody, effectiveToken, effectiveRepoInfo.type, selectedProviderState, selectedModelState, isCustomSelectedModelState, customSelectedModelState, language, branchToUse, modelExcludedDirs, modelExcludedFiles, modelIncludedDirs, modelIncludedFiles, shouldForceReprocess);

      // Use WebSocket for communication
      let responseText = '';

      try {
        // Create WebSocket URL with proper network detection
        const { getWebSocketUrl, getTimeoutConfig, shouldUseWebSocket } = await import('@/utils/networkConfig');
        
        // Only attempt WebSocket in localhost environments where port 8001 is accessible
        // In cloud deployments (Azure, etc.), skip directly to HTTP proxy
        if (!shouldUseWebSocket()) {
          console.log('Cloud environment detected, using HTTP proxy instead of WebSocket');
          throw new Error('Skip WebSocket in cloud environment');
        }
        
        const wsUrl = getWebSocketUrl();
        const timeouts = getTimeoutConfig();
        
        console.log(`Attempting WebSocket connection to: ${wsUrl}`);
        console.log(`Using timeout config:`, timeouts);

        // Create a new WebSocket connection
        const ws = new WebSocket(wsUrl);

        // Set up all event handlers BEFORE waiting for connection
        // This prevents race conditions where messages arrive before handlers are attached
        await new Promise<void>((resolve, reject) => {
          let connectionTimeout: NodeJS.Timeout | null = null;

          // Handle incoming messages
          ws.onmessage = (event) => {
            // Filter out keepalive messages
            const data = event.data;
            if (data && !data.startsWith('<!-- keepalive')) {
              responseText += data;
            }
          };

          // Handle WebSocket close
          ws.onclose = () => {
            console.log('WebSocket connection closed for wiki structure');
            console.log('Total response length:', responseText.length);
            if (connectionTimeout) clearTimeout(connectionTimeout);
            resolve();
          };

          // Handle WebSocket errors
          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (connectionTimeout) clearTimeout(connectionTimeout);
            reject(new Error('WebSocket error'));
          };

          // Handle WebSocket open
          ws.onopen = () => {
            console.log('WebSocket connection established for wiki structure');
            if (connectionTimeout) clearTimeout(connectionTimeout);
            // Send the request as JSON
            ws.send(JSON.stringify(requestBody));
            // Don't resolve here - wait for onclose
          };

          // Set connection timeout
          connectionTimeout = setTimeout(() => {
            console.warn('WebSocket connection timeout');
            ws.close();
            reject(new Error('WebSocket connection timeout'));
          }, 30000); // 30 second timeout for the entire operation
        });
      } catch (wsError) {
        console.error('WebSocket error, falling back to HTTP:', wsError);

        // Fall back to HTTP if WebSocket fails
        const response = await fetch(`/api/chat/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
          throw new Error(`Error determining wiki structure: ${response.status}`);
        }

        // Process the response
        responseText = '';
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('Failed to get response reader');
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          responseText += decoder.decode(value, { stream: true });
        }
      }

      if(responseText.includes('Error preparing retriever: Environment variable OPENAI_API_KEY must be set')) {
         setEmbeddingError(true);
         throw new Error('No embedding provider configured. Please configure one of the following:\n• Azure OpenAI: Set AZURE_OPENAI_EMBEDDING_API_KEY and AZURE_OPENAI_EMBEDDING_ENDPOINT\n• OpenAI: Set OPENAI_API_KEY\n• Or use a local Ollama model for embeddings');
       }

       if(responseText.includes('Ollama model') && responseText.includes('not found')) {
         setEmbeddingError(true);
         throw new Error('The specified Ollama embedding model was not found. Please ensure the model is installed locally or select a different embedding model in the configuration.');
       }

      // Handle content filter warning — strip marker, keep partial content
      // The backend sends [CONTENT_FILTER_WARNING] when finish_reason=content_filter
      let wasContentFiltered = false;
      if (responseText.includes('[CONTENT_FILTER_WARNING]')) {
        console.warn('Wiki structure response was partially truncated by content filter');
        wasContentFiltered = true;
        responseText = responseText.replace(/\n*\[CONTENT_FILTER_WARNING][^]*/m, '');
      }

      // Clean up markdown delimiters
      responseText = responseText.replace(/^```(?:xml)?\s*/i, '').replace(/```\s*$/i, '');

      // Check if response is empty
      if (!responseText || responseText.trim().length === 0) {
        console.error('Empty response received from backend');
        throw new Error('No response received from the backend. The server may have encountered an error during wiki structure generation. Please check the backend logs for details.');
      }

      // Log the response for debugging
      console.log('Wiki structure response length:', responseText.length);
      console.log('Wiki structure response (first 500 chars):', responseText.substring(0, 500));
      console.log('Wiki structure response (last 500 chars):', responseText.substring(responseText.length - 500));

      // Extract wiki structure from response
      let xmlMatch = responseText.match(/<wiki_structure>[\s\S]*?<\/wiki_structure>/m);

      // If XML is incomplete (truncated by content filter), attempt repair
      if (!xmlMatch && responseText.includes('<wiki_structure>')) {
        console.warn('Incomplete wiki_structure XML detected, attempting repair...');
        // Close any open tags so the XML becomes parseable.
        // Strategy: append closing tags for all unclosed elements.
        let repaired = responseText;
        // Collect open tags in order (we need to close them in reverse)
        const openTagStack: string[] = [];
        const tagRegex = /<(\/?)([\w_]+)(?:\s[^>]*)?>/g;
        let m;
        while ((m = tagRegex.exec(repaired)) !== null) {
          const isClosing = m[1] === '/';
          const tagName = m[2];
          if (isClosing) {
            // Pop from stack if matching
            const idx = openTagStack.lastIndexOf(tagName);
            if (idx !== -1) openTagStack.splice(idx, 1);
          } else {
            openTagStack.push(tagName);
          }
        }
        // Close remaining open tags in reverse order
        for (let i = openTagStack.length - 1; i >= 0; i--) {
          repaired += `</${openTagStack[i]}>`;
        }
        xmlMatch = repaired.match(/<wiki_structure>[\s\S]*?<\/wiki_structure>/m);
        if (xmlMatch) {
          console.log('XML repair successful — extracted partial wiki structure');
        } else {
          console.warn('XML repair did not produce a valid wiki_structure block');
        }
      }

      if (!xmlMatch) {
        console.error('Full response text:', responseText);
        // If content was filtered and XML couldn't be parsed/repaired,
        // throw a specific error so the catch block can show actionable guidance
        if (wasContentFiltered) {
          console.warn('Wiki structure failed due to content filter — insufficient XML recovered');
          throw new Error('RETRY_WITH_REDUCED_CONTEXT');
        }
        // Provide a more specific error message based on response content
        const isShortResponse = responseText.trim().length < 200;
        const errorDetail = isShortResponse
          ? 'The response was too short — this may indicate Azure OpenAI content filtering or a model error. Please try again.'
          : 'No valid XML found in response. The response may be incomplete or malformed.';
        throw new Error(errorDetail);
      }

      let xmlText = xmlMatch[0];
      xmlText = xmlText.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
      // Try parsing with DOMParser
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(xmlText, "text/xml");

      // Check for parsing errors
      const parseError = xmlDoc.querySelector('parsererror');
      if (parseError) {
        // Log the first few elements to see what was parsed
        const elements = xmlDoc.querySelectorAll('*');
        if (elements.length > 0) {
          console.log('First 5 element names:',
            Array.from(elements).slice(0, 5).map(el => el.nodeName).join(', '));
        }

        // We'll continue anyway since the XML might still be usable
      }

      // Extract wiki structure
      let title = '';
      let description = '';
      let pages: WikiPage[] = [];

      // Try using DOM parsing first
      const titleEl = xmlDoc.querySelector('title');
      const descriptionEl = xmlDoc.querySelector('description');
      const pagesEls = xmlDoc.querySelectorAll('page');

      title = titleEl ? titleEl.textContent || '' : '';
      description = descriptionEl ? descriptionEl.textContent || '' : '';

      // Parse pages using DOM
      pages = [];

      if (parseError || !pagesEls || pagesEls.length === 0) {
        console.warn('DOM parsing failed or no pages found, trying regex fallback');
        
        // Regex fallback for parsing pages
        const pageRegex = /<page\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/page>/g;
        let pageMatch;
        
        while ((pageMatch = pageRegex.exec(xmlText)) !== null) {
          const pageId = pageMatch[1];
          const pageContent = pageMatch[2];
          
          // Extract title
          const titleMatch = pageContent.match(/<title>([^<]*)<\/title>/);
          const pageTitle = titleMatch ? titleMatch[1] : '';
          
          // Extract importance
          const importanceMatch = pageContent.match(/<importance>([^<]*)<\/importance>/);
          const importance = importanceMatch ? 
            (importanceMatch[1] === 'high' ? 'high' : 
             importanceMatch[1] === 'medium' ? 'medium' : 'low') : 'medium';
          
          // Extract file paths
          const filePaths: string[] = [];
          const filePathRegex = /<file_path>([^<]*)<\/file_path>/g;
          let filePathMatch;
          while ((filePathMatch = filePathRegex.exec(pageContent)) !== null) {
            if (filePathMatch[1]) filePaths.push(filePathMatch[1]);
          }
          
          // Extract related pages
          const relatedPages: string[] = [];
          const relatedRegex = /<related>([^<]*)<\/related>/g;
          let relatedMatch;
          while ((relatedMatch = relatedRegex.exec(pageContent)) !== null) {
            if (relatedMatch[1]) relatedPages.push(relatedMatch[1]);
          }
          
          pages.push({
            id: pageId,
            title: pageTitle,
            content: '',
            filePaths,
            importance: importance as 'high' | 'medium' | 'low',
            relatedPages
          });
        }
        
        console.log('Regex fallback parsed pages:', pages.map(p => p.id));
      } else {
        // DOM parsing succeeded
        pagesEls.forEach(pageEl => {
        const id = pageEl.getAttribute('id') || `page-${pages.length + 1}`;
        const titleEl = pageEl.querySelector('title');
        const importanceEl = pageEl.querySelector('importance');
        const filePathEls = pageEl.querySelectorAll('file_path');
        const relatedEls = pageEl.querySelectorAll('related');

        const title = titleEl ? titleEl.textContent || '' : '';
        const importance = importanceEl ?
          (importanceEl.textContent === 'high' ? 'high' :
            importanceEl.textContent === 'medium' ? 'medium' : 'low') : 'medium';

        const filePaths: string[] = [];
        filePathEls.forEach(el => {
          if (el.textContent) filePaths.push(el.textContent);
        });

        const relatedPages: string[] = [];
        relatedEls.forEach(el => {
          if (el.textContent) relatedPages.push(el.textContent);
        });

        pages.push({
          id,
          title,
          content: '', // Will be generated later
          filePaths,
          importance,
          relatedPages
        });
      });
      } // Close else block for DOM parsing

      // Deduplicate pages by ID — LLM may generate duplicate IDs.
      // Keep the first occurrence; append a suffix to duplicates.
      const seenIds = new Set<string>();
      pages = pages.reduce<WikiPage[]>((acc, page) => {
        if (seenIds.has(page.id)) {
          // Generate a unique ID by appending a suffix
          let newId = page.id;
          let suffix = 2;
          while (seenIds.has(newId)) {
            newId = `${page.id}-${suffix}`;
            suffix++;
          }
          console.warn(`Duplicate page ID "${page.id}" renamed to "${newId}" (title: "${page.title}")`);
          page = { ...page, id: newId };
        }
        seenIds.add(page.id);
        acc.push(page);
        return acc;
      }, []);

      // Extract sections if they exist in the XML
      const sections: WikiSection[] = [];
      const rootSections: string[] = [];

      console.log("Parsed pages with IDs:", pages.map(p => p.id));

      // Try to parse sections if we're in comprehensive view
      if (isComprehensiveView) {
        // Only pick up direct children of <sections>, not nested
        // subsections — querySelectorAll('section') would pick ALL
        const sectionsContainer = xmlDoc.querySelector('sections');
        const topSectionEls = sectionsContainer
          ? Array.from(sectionsContainer.children).filter(
              el => el.tagName.toLowerCase() === 'section'
            )
          : [];
        
        console.log(`Found ${topSectionEls.length} top-level section elements`);

        if (topSectionEls.length > 0) {
          // Recursively parse sections (supports nesting)
          const parseSection = (sectionEl: Element): WikiSection => {
            const id = sectionEl.getAttribute('id') || `section-${sections.length + 1}`;
            const titleEl = sectionEl.querySelector(':scope > title');
            const title = titleEl ? titleEl.textContent || '' : '';

            // Collect page_refs from this section's <pages> element
            const sectionPages: string[] = [];
            const pagesContainer = sectionEl.querySelector(':scope > pages');
            if (pagesContainer) {
              pagesContainer.querySelectorAll('page_ref').forEach(el => {
                if (el.textContent) sectionPages.push(el.textContent);
              });
            }

            // Parse nested subsections
            const subsectionEls = sectionEl.querySelector(':scope > subsections');
            const childSections: WikiSection[] = [];
            if (subsectionEls) {
              Array.from(subsectionEls.children).forEach(child => {
                if (child.tagName.toLowerCase() === 'section') {
                  const childSection = parseSection(child);
                  childSections.push(childSection);
                  // Add child to flat list only if not already present
                  if (!sections.some(s => s.id === childSection.id)) {
                    sections.push(childSection);
                  }
                }
              });
            }

            return {
              id,
              title,
              pages: sectionPages,
              subsections: childSections.length > 0 ? childSections : undefined
            };
          };

          // Parse top-level sections
          topSectionEls.forEach(sectionEl => {
            const section = parseSection(sectionEl);
            if (!sections.some(s => s.id === section.id)) {
              sections.push(section);
            }
            if (!rootSections.includes(section.id)) {
              rootSections.push(section.id);
            }
          });
          
          // Validate: Check for pages not assigned to any section
          const assignedPageIds = new Set<string>();
          sections.forEach(section => {
            section.pages.forEach(pageId => assignedPageIds.add(pageId));
          });
          
          const unassignedPages = pages.filter(page => !assignedPageIds.has(page.id));
          if (unassignedPages.length > 0) {
            console.warn(`Found ${unassignedPages.length} pages not assigned to any section:`, unassignedPages.map(p => p.id));
            
            // Distribute orphaned pages into the most relevant existing section
            // based on their ID prefix (e.g., page "3.4" → section "3")
            // This avoids a catch-all "Additional Topics" bucket.
            for (const orphan of unassignedPages) {
              let bestSection: typeof sections[0] | null = null;

              // Strategy 1: Match by ID prefix (page "3.4" → section "3")
              const idParts = orphan.id.split(/[-.]/).filter(Boolean);
              for (let len = idParts.length - 1; len >= 1; len--) {
                const prefix = idParts.slice(0, len).join('.');
                bestSection = sections.find(s => s.id === prefix) || null;
                if (bestSection) break;
              }

              // Strategy 2: Match by parent_section from page metadata
              if (!bestSection) {
                // The page's filePaths or title might hint at which section it belongs to
                // Fall back to the last section as a reasonable default
                bestSection = sections[sections.length - 1] || null;
              }

              if (bestSection) {
                bestSection.pages.push(orphan.id);
                console.log(`Orphan page "${orphan.id}" (${orphan.title}) → section "${bestSection.id}" (${bestSection.title})`);
              }
            }

            // After distribution, check if any pages are still truly orphaned
            const stillUnassigned = unassignedPages.filter(p => {
              return !sections.some(s => s.pages.includes(p.id));
            });

            if (stillUnassigned.length > 0) {
              // Only create catch-all as absolute last resort
              const additionalSectionId = 'section-additional';
              sections.push({
                id: additionalSectionId,
                title: 'Additional Topics',
                pages: stillUnassigned.map(p => p.id)
              });
              rootSections.push(additionalSectionId);
              console.log(`Created "${additionalSectionId}" for ${stillUnassigned.length} truly orphaned pages`);
            }
          }
        }
      }

      // Create wiki structure
      const wikiStructure: WikiStructure = {
        id: 'wiki',
        title,
        description,
        pages,
        sections,
        rootSections
      };

      // If wiki structure was content-filtered, log and warn about fewer pages
      if (wasContentFiltered) {
        const expectedMin = isComprehensiveView ? 8 : 5;
        if (pages.length < expectedMin) {
          console.warn(
            `Wiki structure was truncated by content filter: got ${pages.length} pages (expected ~${expectedMin}). ` +
            'Proceeding with available pages.'
          );
        }
        // Update description to note truncation
        wikiStructure.description = (wikiStructure.description || '') +
          ' (Note: Wiki structure was partially truncated by content filtering. Some pages may be missing.)';
      }

      setWikiStructure(wikiStructure);
      setCurrentPageId(pages.length > 0 ? pages[0].id : undefined);

      // Start generating content for all pages with controlled concurrency
      if (pages.length > 0) {
        // Mark all pages as in progress
        const initialInProgress = new Set(pages.map(p => p.id));
        setPagesInProgress(initialInProgress);

        logger.info('Starting wiki page generation', { totalPages: pages.length, concurrency: 1 });

        // Maximum concurrent requests
        const MAX_CONCURRENT = 1;
        // Maximum retry attempts for failed pages
        const MAX_RETRIES = 2;
        // Delay between retries (in ms) - exponential backoff
        const RETRY_DELAY_BASE = 5000; // 5 seconds base

        // Create a queue of pages with retry count
        const queue: { page: WikiPage; retries: number }[] = pages.map(p => ({ page: p, retries: 0 }));
        let activeRequests = 0;
        const failedPages: WikiPage[] = [];

        // Function to process next items in queue
        const processQueue = () => {
          // Process as many items as we can up to our concurrency limit
          while (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
            const item = queue.shift();
            if (item) {
              const { page, retries } = item;
              activeRequests++;
              logger.info('Starting page generation', { page: page.title, active: activeRequests, remaining: queue.length, attempt: retries + 1 });

              // Start generating content for this page
              generatePageContent(page, owner, repo)
                .then((result: { success: boolean; error?: string } | void) => {
                  // When done, decrement active count and log with appropriate status
                  activeRequests--;
                  
                  if (result?.success === false) {
                    // Check if we should retry
                    if (retries < MAX_RETRIES) {
                      const retryDelay = RETRY_DELAY_BASE * Math.pow(2, retries); // Exponential backoff
                      logger.warn('Page generation failed, will retry', { 
                        page: page.title, 
                        error: result.error, 
                        attempt: retries + 1,
                        nextAttempt: retries + 2,
                        retryIn: `${retryDelay / 1000}s`
                      });
                      // Add back to queue with incremented retry count after delay
                      setTimeout(() => {
                        queue.push({ page, retries: retries + 1 });
                        if (activeRequests < MAX_CONCURRENT) {
                          processQueue();
                        }
                      }, retryDelay);
                    } else {
                      logger.error('Page generation failed after max retries', { 
                        page: page.title, 
                        error: result.error, 
                        attempts: retries + 1,
                        active: activeRequests, 
                        remaining: queue.length 
                      });
                      failedPages.push(page);
                    }
                  } else {
                    logger.info('Page generation completed', { page: page.title, active: activeRequests, remaining: queue.length });
                  }

                  // Check if all work is done (queue empty and no active requests)
                  if (queue.length === 0 && activeRequests === 0) {
                    if (failedPages.length > 0) {
                      logger.warn('Wiki generation completed with failures', { 
                        failedPages: failedPages.map(p => p.title),
                        failedCount: failedPages.length 
                      });
                    } else {
                      logger.info('All page generation tasks completed successfully');
                    }
                    setIsLoading(false);
                    setLoadingMessage(undefined);
                  } else {
                    // Only process more if there are items remaining and we're under capacity
                    if (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
                      processQueue();
                    }
                  }
                });
            }
          }

          // Additional check: If the queue started empty or becomes empty and no requests were started/active
          if (queue.length === 0 && activeRequests === 0 && pages.length > 0 && pagesInProgress.size === 0) {
            // This handles the case where the queue might finish before the finally blocks fully update activeRequests
            // or if the initial queue was processed very quickly
            console.log("Queue empty and no active requests after loop, ensuring loading is false.");
            setIsLoading(false);
            setLoadingMessage(undefined);
          } else if (pages.length === 0) {
            // Handle case where there were no pages to begin with
            setIsLoading(false);
            setLoadingMessage(undefined);
          }
        };

        // Start processing the queue
        processQueue();

        // Resume queue processing when browser tab becomes visible again.
        // Browsers throttle setTimeout in background tabs, which stalls
        // the page generation pipeline. This listener fires immediately
        // when the user switches back to the tab.
        const onVisibilityChange = () => {
          if (document.visibilityState === 'visible' && queue.length > 0 && activeRequests < MAX_CONCURRENT) {
            console.log('[Wiki] Tab became visible — resuming page generation queue');
            processQueue();
          }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);

        // Clean up listener when all pages are done
        const checkCompletion = setInterval(() => {
          if (queue.length === 0 && activeRequests === 0) {
            document.removeEventListener('visibilitychange', onVisibilityChange);
            clearInterval(checkCompletion);
          }
        }, 5000);
      } else {
        // Set loading to false if there were no pages found
        setIsLoading(false);
        setLoadingMessage(undefined);
      }

    } catch (error) {
      // Content filter retry: show actionable message
      if (error instanceof Error && error.message === 'RETRY_WITH_REDUCED_CONTEXT') {
        console.warn('Wiki structure blocked by content filter');
        setIsLoading(false);
        setError('Wiki structure generation was blocked by Azure content filtering. ' +
          'The repository may contain code patterns (security rules, credentials, firewall configs) ' +
          'that trigger safety checks. Try one of:\n' +
          '• Switch to "Concise" mode (fewer pages, less context)\n' +
          '• Exclude sensitive directories (e.g., Test, Security) via the filter settings\n' +
          '• Click "Refresh Wiki" to retry');
        setLoadingMessage(undefined);
      } else {
        console.error('Error determining wiki structure:', error);
        setIsLoading(false);
        setError(error instanceof Error ? error.message : 'An unknown error occurred');
        setLoadingMessage(undefined);
      }
    } finally {
      setStructureRequestInProgress(false);
    }
  }, [generatePageContent, token, currentToken, effectiveRepoInfo, pagesInProgress.size, structureRequestInProgress, selectedProviderState, selectedModelState, isCustomSelectedModelState, customSelectedModelState, modelExcludedDirs, modelExcludedFiles, language, messages.loading, isComprehensiveView]);

  // Fetch repository structure using GitHub or GitLab API
  const fetchRepositoryStructure = useCallback(async () => {
    // If a request is already in progress, don't start another one
    if (requestInProgress) {
      console.log('Repository fetch already in progress, skipping duplicate call');
      return;
    }

    // Reset previous state
    setWikiStructure(undefined);
    setCurrentPageId(undefined);
    setGeneratedPages({});
    setPagesInProgress(new Set());
    setError(null);
    setPartialCacheMessage(null); // Clear partial cache banner on regeneration
    setEmbeddingError(false); // Reset embedding error state

    try {
      // Set the request in progress flag
      setRequestInProgress(true);

      // MARK: Wiki generation truly starts here (not from complete cache)
      // This enables progress tracking from the very beginning
      setIsGenerationStarted(true);

      // Update loading state
      setIsLoading(true);
      setLoadingMessage(messages.loading?.fetchingStructure || 'Fetching repository structure...');

      let fileTreeData = '';
      let readmeContent = '';
      // Track the detected branch to pass directly to determineWikiStructure
      // This avoids race condition with effectiveRepoInfo state update
      let detectedBranchForWiki: string | null = effectiveRepoInfo.branch || null;

      if (effectiveRepoInfo.type === 'local' && effectiveRepoInfo.localPath) {
        try {
          const response = await fetch(`/local_repo/structure?path=${encodeURIComponent(effectiveRepoInfo.localPath)}`);

          if (!response.ok) {
            const errorData = await response.text();
            throw new Error(`Local repository API error (${response.status}): ${errorData}`);
          }

          const data = await response.json();
          fileTreeData = data.file_tree;
          readmeContent = data.readme;
          // For local repos, we can't determine the actual branch, so use 'main' as default
          setDefaultBranch('main');
          detectedBranchForWiki = 'main';
        } catch (err) {
          throw err;
        }
      } else if (effectiveRepoInfo.type === 'github') {
        // GitHub API approach
        // Try to get the tree data for common branch names
        let treeData = null;
        let apiErrorDetails = '';

        // Determine the GitHub API base URL based on the repository URL
        const getGithubApiUrl = (repoUrl: string | null): string => {
          if (!repoUrl) {
            return 'https://api.github.com'; // Default to public GitHub
          }
          
          try {
            const url = new URL(repoUrl);
            const hostname = url.hostname;
            
            // If it's the public GitHub, use the standard API URL
            if (hostname === 'github.com') {
              return 'https://api.github.com';
            }
            
            // For GitHub Enterprise, use the enterprise API URL format
            // GitHub Enterprise API URL format: https://github.company.com/api/v3
            return `${url.protocol}//${hostname}/api/v3`;
          } catch {
            return 'https://api.github.com'; // Fallback to public GitHub if URL parsing fails
          }
        };

        const githubApiBaseUrl = getGithubApiUrl(effectiveRepoInfo.repoUrl);
        // First, try to get the default branch from the repository info
        let defaultBranchLocal: string | null = null;
        try {
          const repoInfoResponse = await fetch(`${githubApiBaseUrl}/repos/${owner}/${repo}`, {
            headers: createGithubHeaders(currentToken)
          });
          
          if (repoInfoResponse.ok) {
            const repoData = await repoInfoResponse.json();
            defaultBranchLocal = repoData.default_branch;
            console.log(`Found default branch: ${defaultBranchLocal}`);
            // Store the default branch in state
            setDefaultBranch(defaultBranchLocal || 'main');
            detectedBranchForWiki = defaultBranchLocal || 'main';
            // Update effectiveRepoInfo.branch if not explicitly set, so cache uses correct branch name
            if (!effectiveRepoInfo.branch && defaultBranchLocal) {
              setEffectiveRepoInfo(prev => ({ ...prev, branch: defaultBranchLocal }));
            }
          }
        } catch (err) {
          console.warn('Could not fetch repository info for default branch:', err);
        }

        // Create list of branches to try, prioritizing the actual default branch
        const branchesToTry = defaultBranchLocal 
          ? [defaultBranchLocal, 'main', 'master'].filter((branch, index, arr) => arr.indexOf(branch) === index)
          : ['main', 'master'];

        for (const branch of branchesToTry) {
          const apiUrl = `${githubApiBaseUrl}/repos/${owner}/${repo}/git/trees/${branch}?recursive=1`;
          const headers = createGithubHeaders(currentToken);

          console.log(`Fetching repository structure from branch: ${branch}`);
          try {
            const response = await fetch(apiUrl, {
              headers
            });

            if (response.ok) {
              treeData = await response.json();
              console.log('Successfully fetched repository structure');
              break;
            } else {
              const errorData = await response.text();
              apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
              console.error(`Error fetching repository structure: ${apiErrorDetails}`);
            }
          } catch (err) {
            console.error(`Network error fetching branch ${branch}:`, err);
          }
        }

        if (!treeData || !treeData.tree) {
          if (apiErrorDetails) {
            throw new Error(`Could not fetch repository structure. API Error: ${apiErrorDetails}`);
          } else {
            throw new Error('Could not fetch repository structure. Repository might not exist, be empty or private.');
          }
        }

        // Convert tree data to a string representation
        fileTreeData = treeData.tree
          .filter((item: { type: string; path: string }) => item.type === 'blob')
          .map((item: { type: string; path: string }) => item.path)
          .join('\n');

        // Try to fetch README.md content
        try {
          const headers = createGithubHeaders(currentToken);

          const readmeResponse = await fetch(`${githubApiBaseUrl}/repos/${owner}/${repo}/readme`, {
            headers
          });

          if (readmeResponse.ok) {
            const readmeData = await readmeResponse.json();
            readmeContent = atob(readmeData.content);
          } else {
            console.warn(`Could not fetch README.md, status: ${readmeResponse.status}`);
          }
        } catch (err) {
          console.warn('Could not fetch README.md, continuing with empty README', err);
        }
      }
      else if (effectiveRepoInfo.type === 'gitlab') {
        // GitLab API approach
        const projectPath = extractUrlPath(effectiveRepoInfo.repoUrl ?? '')?.replace(/\.git$/, '') || `${owner}/${repo}`;
        const projectDomain = extractUrlDomain(effectiveRepoInfo.repoUrl ?? "https://gitlab.com");
        const encodedProjectPath = encodeURIComponent(projectPath);

        const headers = createGitlabHeaders(currentToken);

        interface GitLabFile {
          id: string;
          name: string;
          type: 'tree' | 'blob';
          path: string;
          mode: string;
        }
        const filesData: GitLabFile[] = [];

        try {
          // Step 1: Get project info to determine default branch
          let projectInfoUrl: string;
          let defaultBranchLocal = 'main'; // fallback
          try {
            const validatedUrl = new URL(projectDomain ?? ''); // Validate domain
            projectInfoUrl = `${validatedUrl.origin}/api/v4/projects/${encodedProjectPath}`;
          } catch (err) {
            throw new Error(`Invalid project domain URL: ${projectDomain}`);
          }
          const projectInfoRes = await fetch(projectInfoUrl, { headers });

          if (!projectInfoRes.ok) {
            const errorData = await projectInfoRes.text();
            throw new Error(`GitLab project info error: Status ${projectInfoRes.status}, Response: ${errorData}`);
          }

          const projectInfo = await projectInfoRes.json();
          defaultBranchLocal = projectInfo.default_branch || 'main';
          console.log(`Found GitLab default branch: ${defaultBranchLocal}`);
          // Store the default branch in state
          setDefaultBranch(defaultBranchLocal);
          detectedBranchForWiki = defaultBranchLocal;
          // Update effectiveRepoInfo.branch if not explicitly set, so cache uses correct branch name
          if (!effectiveRepoInfo.branch && defaultBranchLocal) {
            setEffectiveRepoInfo(prev => ({ ...prev, branch: defaultBranchLocal }));
          }

          // Step 2: Paginate to fetch full file tree
          let page = 1;
          let morePages = true;
          
          while (morePages) {
            const apiUrl = `${projectInfoUrl}/repository/tree?recursive=true&per_page=100&page=${page}`;
            const response = await fetch(apiUrl, { headers });

            if (!response.ok) {
                const errorData = await response.text();
              throw new Error(`Error fetching GitLab repository structure (page ${page}): ${errorData}`);
            }

            const pageData = await response.json();
            filesData.push(...pageData);

            const nextPage = response.headers.get('x-next-page');
            morePages = !!nextPage;
            page = nextPage ? parseInt(nextPage, 10) : page + 1;
        }

          if (!Array.isArray(filesData) || filesData.length === 0) {
            throw new Error('Could not fetch repository structure. Repository might be empty or inaccessible.');
        }

          // Step 3: Format file paths
        fileTreeData = filesData
          .filter((item: { type: string; path: string }) => item.type === 'blob')
          .map((item: { type: string; path: string }) => item.path)
          .join('\n');

          // Step 4: Try to fetch README.md content
          const readmeUrl = `${projectInfoUrl}/repository/files/README.md/raw`;
            try {
            const readmeResponse = await fetch(readmeUrl, { headers });
              if (readmeResponse.ok) {
                readmeContent = await readmeResponse.text();
                console.log('Successfully fetched GitLab README.md');
              } else {
              console.warn(`Could not fetch GitLab README.md status: ${readmeResponse.status}`);
              }
            } catch (err) {
            console.warn(`Error fetching GitLab README.md:`, err);
            }
        } catch (err) {
          console.error("Error during GitLab repository tree retrieval:", err);
          throw err;
        }
      }
      else if (effectiveRepoInfo.type === 'bitbucket') {
        // Bitbucket API approach
        const repoPath = extractUrlPath(effectiveRepoInfo.repoUrl ?? '') ?? `${owner}/${repo}`;
        const encodedRepoPath = encodeURIComponent(repoPath);

        // Try to get the file tree for common branch names
        let filesData = null;
        let apiErrorDetails = '';
        let defaultBranchLocal = '';
        const headers = createBitbucketHeaders(currentToken);

        // First get project info to determine default branch
        const projectInfoUrl = `https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}`;
        try {
          const response = await fetch(projectInfoUrl, { headers });

          const responseText = await response.text();

          if (response.ok) {
            const projectData = JSON.parse(responseText);
            defaultBranchLocal = projectData.mainbranch.name;
            // Store the default branch in state
            setDefaultBranch(defaultBranchLocal);
            detectedBranchForWiki = defaultBranchLocal;

            const apiUrl = `https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}/src/${defaultBranchLocal}/?recursive=true&per_page=100`;
            try {
              const response = await fetch(apiUrl, {
                headers
              });

              const structureResponseText = await response.text();

              if (response.ok) {
                filesData = JSON.parse(structureResponseText);
              } else {
                const errorData = structureResponseText;
                apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
              }
            } catch (err) {
              console.error(`Network error fetching Bitbucket branch ${defaultBranchLocal}:`, err);
            }
          } else {
            const errorData = responseText;
            apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
          }
        } catch (err) {
          console.error("Network error fetching Bitbucket project info:", err);
        }

        if (!filesData || !Array.isArray(filesData.values) || filesData.values.length === 0) {
          if (apiErrorDetails) {
            throw new Error(`Could not fetch repository structure. Bitbucket API Error: ${apiErrorDetails}`);
          } else {
            throw new Error('Could not fetch repository structure. Repository might not exist, be empty or private.');
          }
        }

        // Convert files data to a string representation
        fileTreeData = filesData.values
          .filter((item: { type: string; path: string }) => item.type === 'commit_file')
          .map((item: { type: string; path: string }) => item.path)
          .join('\n');

        // Try to fetch README.md content
        try {
          const headers = createBitbucketHeaders(currentToken);

          const readmeResponse = await fetch(`https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}/src/${defaultBranchLocal}/README.md`, {
            headers
          });

          if (readmeResponse.ok) {
            readmeContent = await readmeResponse.text();
          } else {
            console.warn(`Could not fetch Bitbucket README.md, status: ${readmeResponse.status}`);
          }
        } catch (err) {
          console.warn('Could not fetch Bitbucket README.md, continuing with empty README', err);
        }
      }
      else if (effectiveRepoInfo.type === 'azuredevops') {
        // Azure DevOps API approach
        try {
          setLoadingMessage(messages.loading?.fetchingStructure || 'Fetching Azure DevOps repository structure...');

          // Use tokenRef for synchronous access (avoids React state update race conditions)
          // Fall back to state values for backward compatibility
          const effectiveToken = tokenRef.current || token || currentToken;
          console.log('[AzureDevOps] Calling structure API with:', {
            repoUrl: effectiveRepoInfo.repoUrl,
            hasToken: !!effectiveToken,
            tokenLength: effectiveToken?.length || 0,
            usingTokenRef: !!tokenRef.current,
            usingTokenState: !!token,
            usingCurrentToken: !!currentToken
          });

          // Use the data pipeline API for Azure DevOps repositories
          const response = await fetch('/api/azure-devops/structure', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              repo_url: effectiveRepoInfo.repoUrl,
              token: effectiveToken
            })
          });

          if (!response.ok) {
            const errorText = await response.text().catch(() => 'No error details available');
            throw new Error(`Azure DevOps API error (${response.status}): ${errorText}`);
          }

          const data = await response.json();
          fileTreeData = data.file_tree || '';
          readmeContent = data.readme || '';
          
          // Store the default branch in state (Azure DevOps typically uses 'main' or 'master')
          const detectedBranch = data.default_branch || 'main';
          setDefaultBranch(detectedBranch);
          detectedBranchForWiki = detectedBranch;
          // Update effectiveRepoInfo.branch if not explicitly set, so cache uses correct branch name
          if (!effectiveRepoInfo.branch && detectedBranch) {
            setEffectiveRepoInfo(prev => ({ ...prev, branch: detectedBranch }));
          }

          if (!fileTreeData) {
            throw new Error('Could not fetch repository structure. Repository might not exist, be empty or private. Please check your Personal Access Token (PAT).');
          }
        } catch (err) {
          console.error('Error fetching Azure DevOps repository structure:', err);
          throw err;
        }
      }

      // Now determine the wiki structure
      // Pass detectedBranchForWiki directly to avoid race condition with effectiveRepoInfo state update
      await determineWikiStructure(fileTreeData, readmeContent, owner, repo, detectedBranchForWiki);

    } catch (error) {
      console.error('Error fetching repository structure:', error);
      setIsLoading(false);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      setLoadingMessage(undefined);
    } finally {
      // Reset the request in progress flag
      setRequestInProgress(false);
    }
  }, [owner, repo, determineWikiStructure, currentToken, token, effectiveRepoInfo, requestInProgress, messages.loading]);

  // Function to export wiki content
  const exportWiki = useCallback(async (format: 'markdown' | 'json') => {
    if (!wikiStructure || Object.keys(generatedPages).length === 0) {
      setExportError('No wiki content to export');
      return;
    }

    try {
      setIsExporting(true);
      setExportError(null);
      setLoadingMessage(`${language === 'ja' ? 'Wikiを' : 'Exporting wiki as '} ${format} ${language === 'ja' ? 'としてエクスポート中...' : '...'}`);

      // Prepare the pages for export
      const pagesToExport = wikiStructure.pages.map(page => {
        // Use the generated content if available, otherwise use an empty string
        const content = generatedPages[page.id]?.content || 'Content not generated';
        return {
          ...page,
          content
        };
      });

      // Get repository URL
      const repoUrl = getRepoUrl(effectiveRepoInfo);

      // Make API call to export wiki
      const response = await fetch(`/export/wiki`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: repoUrl,
          type: effectiveRepoInfo.type,
          pages: pagesToExport,
          format
        })
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error details available');
        throw new Error(`Error exporting wiki: ${response.status} - ${errorText}`);
      }

      // Get the filename from the Content-Disposition header if available
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${effectiveRepoInfo.repo}_wiki.${format === 'markdown' ? 'md' : 'json'}`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }

      // Convert the response to a blob and download it
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

    } catch (err) {
      console.error('Error exporting wiki:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error during export';
      setExportError(errorMessage);
    } finally {
      setIsExporting(false);
      setLoadingMessage(undefined);
    }
  }, [wikiStructure, generatedPages, effectiveRepoInfo, language]);

  // No longer needed as we use the modal directly

  const confirmRefresh = useCallback(async (newToken?: string, newComprehensiveValue?: boolean) => {
    setShowModelOptions(false);
    setLoadingMessage(messages.loading?.clearingCache || 'Clearing server cache...');
    setIsLoading(true); // Show loading indicator immediately

    // Use the new comprehensive value passed from modal (avoids React state timing issues)
    const targetComprehensive = newComprehensiveValue ?? isComprehensiveView;

    try {
      const params = new URLSearchParams({
        owner: effectiveRepoInfo.owner,
        repo: effectiveRepoInfo.repo,
        repo_type: effectiveRepoInfo.type,
        language: language,
        provider: selectedProviderState,
        model: selectedModelState,
        is_custom_model: isCustomSelectedModelState.toString(),
        custom_model: customSelectedModelState,
        comprehensive: targetComprehensive.toString(),
        authorization_code: authCode,
      });

      // Add branch parameter if available
      if (effectiveRepoInfo.branch) {
        params.append('branch', effectiveRepoInfo.branch);
      }

      // Add file filters configuration
      if (modelExcludedDirs) {
        params.append('excluded_dirs', modelExcludedDirs);
      }
      if (modelExcludedFiles) {
        params.append('excluded_files', modelExcludedFiles);
      }

      if(authRequired && !authCode) {
        setIsLoading(false);
        console.error("Authorization code is required");
        setError('Authorization code is required');
        return;
      }

      // Add timeout to prevent hanging when backend is slow/unavailable
      const deleteController = new AbortController();
      const deleteTimeout = setTimeout(() => deleteController.abort(), 15000); // 15 second timeout
      
      const response = await fetch(`/api/wiki_cache?${params.toString()}`, {
        method: 'DELETE',
        headers: {
          'Accept': 'application/json',
        },
        signal: deleteController.signal
      });
      clearTimeout(deleteTimeout);

      if (response.ok) {
        console.log('Server-side wiki cache cleared successfully.');
        // Optionally, show a success message for cache clearing if desired
        // setLoadingMessage('Cache cleared. Refreshing wiki...');
      } else {
        const errorText = await response.text();
        console.warn(`Failed to clear server-side wiki cache (status: ${response.status}): ${errorText}. Proceeding with refresh anyway.`);
        // Optionally, inform the user about the cache clear failure but that refresh will still attempt
        // setError(\`Cache clear failed: ${errorText}. Trying to refresh...\`);
        if(response.status == 401) {
          setIsLoading(false);
          setLoadingMessage(undefined);
          setError('Failed to validate the authorization code');
          console.error('Failed to validate the authorization code')
          return;
        }
      }
    } catch (err) {
      console.warn('Error calling DELETE /api/wiki_cache:', err);
      setIsLoading(false);
      setEmbeddingError(false); // Reset embedding error state
      // Optionally, inform the user about the cache clear error
      // setError(\`Error clearing cache: ${err instanceof Error ? err.message : String(err)}. Trying to refresh...\`);
      throw err;
    }

    // Update token if provided
    if (newToken) {
      // Update token ref synchronously (avoids race conditions)
      tokenRef.current = newToken;
      // Update current token state
      setCurrentToken(newToken);
      // Also update the token state to trigger dependent useEffects
      setToken(newToken);
      // Store in sessionStorage for persistence
      const tokenKey = `deepwiki_token_${effectiveRepoInfo.owner}_${effectiveRepoInfo.repo}`;
      sessionStorage.setItem(tokenKey, newToken);
      console.log('[Token] Token updated via settings modal');
    }

    // Proceed with the rest of the refresh logic
    console.log('Refreshing wiki. Server cache will be overwritten upon new generation if not cleared.');

    // Clear the localStorage cache (if any remnants or if it was used before this change)
    // Use targetComprehensive to clear the correct mode's cache
    const localStorageCacheKey = getCacheKey(effectiveRepoInfo.owner, effectiveRepoInfo.repo, effectiveRepoInfo.type, language, targetComprehensive);
    localStorage.removeItem(localStorageCacheKey);

    // Reset cache loaded flag
    cacheLoadedSuccessfully.current = false;
    effectRan.current = false; // Allow the main data loading useEffect to run again

    // Set force_reprocess flag to trigger migration from pkl to vector-based storage
    // This flag is used once during wiki structure determination and then reset
    forceReprocessRef.current = true;

    // Reset all state
    setWikiStructure(undefined);
    setCurrentPageId(undefined);
    setGeneratedPages({});
    setPagesInProgress(new Set());
    setError(null);
    setEmbeddingError(false); // Reset embedding error state
    setIsGenerationStarted(false); // Will be set when fetchRepositoryStructure is called
    setIsLoading(true); // Set loading state for refresh
    setLoadingMessage(messages.loading?.initializing || 'Initializing wiki generation...');

    // Clear any in-progress requests for page content
    activeContentRequests.clear();
    // Reset flags related to request processing if they are component-wide
    setStructureRequestInProgress(false); // Assuming this flag should be reset
    setRequestInProgress(false); // Assuming this flag should be reset

    // Explicitly trigger the data loading process again by re-invoking what the main useEffect does.
    // This will first attempt to load from (now hopefully non-existent or soon-to-be-overwritten) server cache,
    // then proceed to fetchRepositoryStructure if needed.
    // To ensure fetchRepositoryStructure is called if cache is somehow still there or to force a full refresh:
    // One option is to directly call fetchRepositoryStructure() if force refresh means bypassing cache check.
    // For now, we rely on the standard loadData flow initiated by resetting effectRan and dependencies.
    // This will re-trigger the main data loading useEffect.
    // No direct call to fetchRepositoryStructure here, let the useEffect handle it based on effectRan.current = false.
  }, [effectiveRepoInfo, language, messages.loading, activeContentRequests, selectedProviderState, selectedModelState, isCustomSelectedModelState, customSelectedModelState, modelExcludedDirs, modelExcludedFiles, isComprehensiveView, authCode, authRequired]);

  // Start wiki generation when component mounts
  useEffect(() => {
    // Wait for token to be checked from sessionStorage before proceeding
    // This prevents race condition where Azure DevOps check fails before token is loaded
    if (!tokenChecked) {
      logger.debug('Waiting for token check to complete before wiki init');
      return;
    }

    // Clear paused state if resuming (user navigated back to this page)
    const currentProgress = progressRef.current;
    if (currentProgress && 
        currentProgress.owner === owner && 
        currentProgress.repo === repo && 
        currentProgress.isPaused) {
      console.log('[WikiPage] Clearing paused state for resumed wiki');
      setProgress({
        ...currentProgress,
        isPaused: false,
      });
    }

    if (effectRan.current === false) {
      effectRan.current = true; // Set to true immediately to prevent re-entry due to StrictMode

      const loadData = async () => {
        // Try loading from server-side cache first (no token needed for cache)
        setLoadingMessage(messages.loading?.fetchingCache || 'Checking for cached wiki...');
        try {
          const params = new URLSearchParams({
            owner: effectiveRepoInfo.owner,
            repo: effectiveRepoInfo.repo,
            repo_type: effectiveRepoInfo.type,
            language: language,
            comprehensive: isComprehensiveView.toString(),
          });

          // Add branch parameter if available
          if (effectiveRepoInfo.branch) {
            params.append('branch', effectiveRepoInfo.branch);
          }
          
          // Add timeout to prevent hanging when backend is slow/unavailable
          const cacheController = new AbortController();
          const cacheTimeout = setTimeout(() => cacheController.abort(), 10000); // 10 second timeout
          
          const response = await fetch(`/api/wiki_cache?${params.toString()}`, {
            signal: cacheController.signal
          });
          clearTimeout(cacheTimeout);

          if (response.ok) {
            const cachedData = await response.json(); // Returns null if no cache
            if (cachedData && cachedData.wiki_structure && cachedData.generated_pages && Object.keys(cachedData.generated_pages).length > 0) {
              // Check if this is a partial/incomplete cache
              const totalPages = cachedData.wiki_structure.pages?.length || 0;
              const cachedPagesCount = Object.keys(cachedData.generated_pages).length;
              const pagesWithContent = (Object.values(cachedData.generated_pages) as WikiPage[]).filter(
                (p: WikiPage) => p.content && p.content !== 'Loading...' && !p.content.startsWith('Error')
              ).length;
              const isPartialCache = cachedData.is_partial === true || pagesWithContent < totalPages;
              
              if (isPartialCache) {
                console.log(`Found partial cache: ${pagesWithContent}/${totalPages} pages with content`);
              } else {
                console.log('Using complete server-cached wiki data');
              }
              
              if(cachedData.model) {
                setSelectedModelState(cachedData.model);
              }
              if(cachedData.provider) {
                setSelectedProviderState(cachedData.provider);
              }

              // Update repoInfo
              if(cachedData.repo) {
                setEffectiveRepoInfo(cachedData.repo);
              } else if (cachedData.repo_url && !effectiveRepoInfo.repoUrl) {
                const updatedRepoInfo = { ...effectiveRepoInfo, repoUrl: cachedData.repo_url };
                setEffectiveRepoInfo(updatedRepoInfo); // Update effective repo info state
                console.log('Using cached repo_url:', cachedData.repo_url);
              }

              // Ensure the cached structure has sections and rootSections
              const cachedStructure = {
                ...cachedData.wiki_structure,
                sections: cachedData.wiki_structure.sections || [],
                rootSections: cachedData.wiki_structure.rootSections || []
              };

              // If sections or rootSections are missing, create intelligent ones based on page titles
              if (!cachedStructure.sections.length || !cachedStructure.rootSections.length) {
                const pages = cachedStructure.pages;
                const sections: WikiSection[] = [];
                const rootSections: string[] = [];

                // Group pages by common prefixes or categories
                const pageClusters = new Map<string, WikiPage[]>();

                // Define common categories that might appear in page titles
                const categories = [
                  { id: 'overview', title: 'Overview', keywords: ['overview', 'introduction', 'about'] },
                  { id: 'architecture', title: 'Architecture', keywords: ['architecture', 'structure', 'design', 'system'] },
                  { id: 'features', title: 'Core Features', keywords: ['feature', 'functionality', 'core'] },
                  { id: 'components', title: 'Components', keywords: ['component', 'module', 'widget'] },
                  { id: 'api', title: 'API', keywords: ['api', 'endpoint', 'service', 'server'] },
                  { id: 'data', title: 'Data Flow', keywords: ['data', 'flow', 'pipeline', 'storage'] },
                  { id: 'models', title: 'Models', keywords: ['model', 'ai', 'ml', 'integration'] },
                  { id: 'ui', title: 'User Interface', keywords: ['ui', 'interface', 'frontend', 'page'] },
                  { id: 'setup', title: 'Setup & Configuration', keywords: ['setup', 'config', 'installation', 'deploy'] }
                ];

                // Initialize clusters with empty arrays
                categories.forEach(category => {
                  pageClusters.set(category.id, []);
                });

                // Add an "Other" category for pages that don't match any category
                pageClusters.set('other', []);

                // Assign pages to categories based on title keywords
                pages.forEach((page: WikiPage) => {
                  const title = page.title.toLowerCase();
                  let assigned = false;

                  // Try to find a matching category
                  for (const category of categories) {
                    if (category.keywords.some(keyword => title.includes(keyword))) {
                      pageClusters.get(category.id)?.push(page);
                      assigned = true;
                      break;
                    }
                  }

                  // If no category matched, put in "Other"
                  if (!assigned) {
                    pageClusters.get('other')?.push(page);
                  }
                });

                // Create sections for non-empty categories
                for (const [categoryId, categoryPages] of pageClusters.entries()) {
                  if (categoryPages.length > 0) {
                    const category = categories.find(c => c.id === categoryId) ||
                                    { id: categoryId, title: categoryId === 'other' ? 'Other' : categoryId.charAt(0).toUpperCase() + categoryId.slice(1) };

                    const sectionId = `section-${categoryId}`;
                    sections.push({
                      id: sectionId,
                      title: category.title,
                      pages: categoryPages.map((p: WikiPage) => p.id)
                    });
                    rootSections.push(sectionId);

                    // Update page parentId
                    categoryPages.forEach((page: WikiPage) => {
                      page.parentId = sectionId;
                    });
                  }
                }

                // If we still have no sections (unlikely), fall back to importance-based grouping
                if (sections.length === 0) {
                  const highImportancePages = pages.filter((p: WikiPage) => p.importance === 'high').map((p: WikiPage) => p.id);
                  const mediumImportancePages = pages.filter((p: WikiPage) => p.importance === 'medium').map((p: WikiPage) => p.id);
                  const lowImportancePages = pages.filter((p: WikiPage) => p.importance === 'low').map((p: WikiPage) => p.id);

                  if (highImportancePages.length > 0) {
                    sections.push({
                      id: 'section-high',
                      title: 'Core Components',
                      pages: highImportancePages
                    });
                    rootSections.push('section-high');
                  }

                  if (mediumImportancePages.length > 0) {
                    sections.push({
                      id: 'section-medium',
                      title: 'Key Features',
                      pages: mediumImportancePages
                    });
                    rootSections.push('section-medium');
                  }

                  if (lowImportancePages.length > 0) {
                    sections.push({
                      id: 'section-low',
                      title: 'Additional Information',
                      pages: lowImportancePages
                    });
                    rootSections.push('section-low');
                  }
                }

                cachedStructure.sections = sections;
                cachedStructure.rootSections = rootSections;
              }

              // Validate: Check for pages not assigned to any section and add them to "Additional Topics"
              const assignedPageIds = new Set<string>();
              cachedStructure.sections.forEach((section: WikiSection) => {
                section.pages.forEach((pageId: string) => assignedPageIds.add(pageId));
              });
              
              const unassignedPages = cachedStructure.pages.filter((page: WikiPage) => !assignedPageIds.has(page.id));
              if (unassignedPages.length > 0) {
                console.warn(`Found ${unassignedPages.length} pages not assigned to any section:`, unassignedPages.map((p: WikiPage) => p.id));
                
                // Create an "Additional Topics" section for orphaned pages
                const additionalSectionId = 'section-additional';
                cachedStructure.sections.push({
                  id: additionalSectionId,
                  title: 'Additional Topics',
                  pages: unassignedPages.map((p: WikiPage) => p.id)
                });
                cachedStructure.rootSections.push(additionalSectionId);
                console.log(`Created "${additionalSectionId}" section for unassigned pages`);
              }

              setWikiStructure(cachedStructure);
              setGeneratedPages(cachedData.generated_pages);
              setCurrentPageId(cachedStructure.pages.length > 0 ? cachedStructure.pages[0].id : undefined);
              
              // If partial cache, set up for resumption and continue generation
              if (isPartialCache) {
                console.log('Partial cache detected - will resume generation for missing pages');
                setIsResumingFromPartial(true);
                cacheLoadedSuccessfully.current = false; // Allow checkpoints to be saved
                
                // For Azure DevOps without a token, display partial cache but don't try to resume
                const effectiveTokenForPartial = token || currentToken;
                if (effectiveRepoInfo.type === 'azuredevops' && !effectiveTokenForPartial) {
                  logger.info('Azure DevOps partial cache displayed without token - cannot resume generation', {
                    pagesWithContent,
                    totalPages
                  });
                  setIsLoading(false);
                  setEmbeddingError(false);
                  setLoadingMessage(undefined);
                  // Show informative banner (not blocking error) so wiki content is still displayed
                  setPartialCacheMessage(`Partial wiki displayed (${pagesWithContent}/${totalPages} pages). To generate remaining pages, please provide a Personal Access Token (PAT) via the Settings button.`);
                  cacheLoadedSuccessfully.current = true; // Treat as successfully loaded (partial)
                  return; // Display partial cache without trying to resume
                }
                
                // Don't set isLoading to false - continue to generate missing pages
                setLoadingMessage(`Resuming wiki generation (${pagesWithContent}/${totalPages} pages cached)...`);
                
                // Identify pages that need to be generated
                const pagesToGenerate = cachedStructure.pages.filter((page: WikiPage) => {
                  const cachedPage = cachedData.generated_pages[page.id];
                  return !cachedPage || !cachedPage.content || cachedPage.content === 'Loading...' || cachedPage.content.startsWith('Error');
                });
                
                console.log(`[Partial Cache Resume] ${pagesToGenerate.length} pages need to be generated`);
                
                // Mark generation as started for progress tracking (partial cache needs generation)
                setIsGenerationStarted(true);
                
                // Start generating missing pages using the existing generation queue
                if (pagesToGenerate.length > 0) {
                  // Use the existing page generation logic (which will be triggered by the normal flow)
                  // Just need to ensure the pages are marked for generation
                  console.log('[Partial Cache Resume] Missing pages will be generated:', pagesToGenerate.map((p: WikiPage) => p.title));
                  
                  // Mark pages as in progress for progress bar display
                  setPagesInProgress(new Set(pagesToGenerate.map((p: WikiPage) => p.id)));
                  
                  // Process pages with controlled concurrency (same as normal flow)
                  const MAX_CONCURRENT = 1;
                  const MAX_RETRIES = 2;
                  const RETRY_DELAY_BASE = 5000;
                  
                  const queue: { page: WikiPage; retries: number }[] = pagesToGenerate.map((p: WikiPage) => ({ page: p, retries: 0 }));
                  let activeRequests = 0;
                  const failedPages: WikiPage[] = [];
                  
                  const processQueue = () => {
                    while (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
                      const item = queue.shift();
                      if (item) {
                        const { page, retries } = item;
                        activeRequests++;
                        console.log(`[Partial Cache Resume] Generating page: ${page.title}, active: ${activeRequests}, remaining: ${queue.length}`);
                        
                        generatePageContent(page, owner, repo)
                          .then((result: { success: boolean; error?: string } | void) => {
                            activeRequests--;
                            
                            if (result?.success === false) {
                              if (retries < MAX_RETRIES) {
                                const retryDelay = RETRY_DELAY_BASE * Math.pow(2, retries);
                                console.warn(`[Partial Cache Resume] Page ${page.title} failed, retrying in ${retryDelay / 1000}s`);
                                setTimeout(() => {
                                  queue.push({ page, retries: retries + 1 });
                                  if (activeRequests < MAX_CONCURRENT) {
                                    processQueue();
                                  }
                                }, retryDelay);
                              } else {
                                console.error(`[Partial Cache Resume] Page ${page.title} failed after max retries`);
                                failedPages.push(page);
                              }
                            }
                            
                            // Check if all work is done
                            if (queue.length === 0 && activeRequests === 0) {
                              if (failedPages.length > 0) {
                                console.warn(`[Partial Cache Resume] Completed with ${failedPages.length} failures`);
                              } else {
                                console.log('[Partial Cache Resume] All pages generated successfully');
                              }
                              setIsLoading(false);
                              setLoadingMessage(undefined);
                            } else if (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
                              processQueue();
                            }
                          });
                      }
                    }
                  };
                  
                  // Start processing
                  processQueue();
                } else {
                  // No pages to generate (shouldn't happen for partial cache, but handle it)
                  setIsLoading(false);
                  setLoadingMessage(undefined);
                }
                
                cacheLoadedSuccessfully.current = false; // Allow checkpoints to be saved
                return; // Don't call fetchRepositoryStructure for partial cache
              } else {
                // Complete cache - just display it
                setIsLoading(false);
                setIsGenerationStarted(false); // No generation needed for complete cache
                setEmbeddingError(false); 
                setLoadingMessage(undefined);
                cacheLoadedSuccessfully.current = true;
                return; // Exit if cache is successfully loaded
              }
            } else {
              logger.info('No valid wiki data in server cache or cache is empty');
            }
          } else {
            // Log error but proceed to fetch structure, as cache is optional
            logger.error('Error fetching wiki cache from server', { status: response.status });
          }
        } catch (error) {
          logger.error('Error loading from server cache', { error: String(error) });
          // Proceed to fetch structure if cache loading fails
        }

        // If we reached here, either there was no cache, it was invalid, or an error occurred
        // For Azure DevOps, we need a token to fetch from the API
        // Use tokenRef for synchronous access (avoids React state update race conditions)
        const effectiveToken = tokenRef.current || token || currentToken;
        if (effectiveRepoInfo.type === 'azuredevops' && !effectiveToken) {
          logger.warn('Azure DevOps repo requires PAT but no token available', { 
            owner: effectiveRepoInfo.owner, 
            repo: effectiveRepoInfo.repo,
            hasTokenRef: !!tokenRef.current,
            hasToken: !!token,
            hasCurrentToken: !!currentToken
          });
          // Stop loading and show error prompting for token
          setIsLoading(false);
          setLoadingMessage(undefined);
          setError('Azure DevOps repositories require a Personal Access Token (PAT) to generate wiki. Please click the Settings button below to provide your PAT.');
          // Don't reset effectRan - user needs to provide token first via the modal
          return;
        }
        
        // Proceed to fetch repository structure
        fetchRepositoryStructure();
      };

      loadData();

    } else {
      logger.debug('Skipping duplicate repository fetch/cache check');
    }

    // Clean up function for this effect is not strictly necessary for loadData,
    // but keeping the main unmount cleanup in the other useEffect
  }, [effectiveRepoInfo, effectiveRepoInfo.owner, effectiveRepoInfo.repo, effectiveRepoInfo.type, language, fetchRepositoryStructure, messages.loading?.fetchingCache, isComprehensiveView, token, currentToken, tokenChecked, owner, repo]);
  // Note: progress and wikiStructure deliberately excluded to prevent re-triggering when resuming from cache

  // Save wiki to server-side cache when generation is complete
  useEffect(() => {
    const saveCache = async () => {
      if (!isLoading &&
          !error &&
          wikiStructure &&
          Object.keys(generatedPages).length > 0 &&
          Object.keys(generatedPages).length >= wikiStructure.pages.length &&
          (!cacheLoadedSuccessfully.current || isResumingFromPartial)) {

        const allPagesHaveContent = wikiStructure.pages.every(page =>
          generatedPages[page.id] && generatedPages[page.id].content && generatedPages[page.id].content !== 'Loading...');

        if (allPagesHaveContent) {
          console.log('Attempting to save COMPLETE wiki data to server cache via Next.js proxy');

          try {
            // Make sure wikiStructure has sections and rootSections
            const structureToCache = {
              ...wikiStructure,
              sections: wikiStructure.sections || [],
              rootSections: wikiStructure.rootSections || []
            };
            const dataToCache = {
              repo: effectiveRepoInfo,
              language: language,
              comprehensive: isComprehensiveView,
              wiki_structure: structureToCache,
              generated_pages: generatedPages,
              provider: selectedProviderState,
              model: selectedModelState,
              is_partial: false  // Mark as complete cache
            };
            
            // Add timeout to prevent hanging when backend is slow/unavailable
            const saveController = new AbortController();
            const saveTimeout = setTimeout(() => saveController.abort(), 30000); // 30 second timeout for POST (larger payload)
            
            const response = await fetch(`/api/wiki_cache`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(dataToCache),
              signal: saveController.signal
            });
            clearTimeout(saveTimeout);

            if (response.ok) {
              console.log('Wiki data successfully saved to server cache (complete)');
              // Reset flags after successful complete save
              cacheLoadedSuccessfully.current = true;
              setIsResumingFromPartial(false);
            } else {
              console.error('Error saving wiki data to server cache:', response.status, await response.text());
            }
          } catch (error) {
            console.error('Error saving to server cache:', error);
          }
        }
      }
    };

    saveCache();
  }, [isLoading, error, wikiStructure, generatedPages, effectiveRepoInfo.owner, effectiveRepoInfo.repo, effectiveRepoInfo.type, effectiveRepoInfo.repoUrl, repoUrl, language, isComprehensiveView, isResumingFromPartial]);

  const handlePageSelect = (pageId: string) => {
    if (currentPageId != pageId) {
      setCurrentPageId(pageId)
    }
  };

  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);

  return (
    <div className="h-screen paper-texture p-4 md:p-8 flex flex-col">
      <style>{wikiStyles}</style>

      <header className="max-w-[90%] xl:max-w-[1400px] mx-auto mb-8 h-fit w-full">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center gap-1.5 transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5">
              <FaHome /> {messages.repoPage?.home || 'Home'}
            </Link>
          </div>
        </div>
      </header>

      <main className={`flex-1 mx-auto overflow-hidden ${wikiStructure && !isChatPanelCollapsed ? 'w-full px-4' : 'max-w-[90%] xl:max-w-[1400px]'}`}>
        {isLoading && !isMinimized ? (
          <div className="flex flex-col items-center justify-center p-8 bg-[var(--card-bg)] rounded shadow-custom card-azure max-w-2xl mx-auto">
            <div className="relative mb-6">
              <div className="absolute -inset-4 bg-[var(--accent-primary)]/10 rounded-full blur-md animate-pulse"></div>
              <div className="relative flex items-center justify-center">
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse"></div>
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-75 mx-2"></div>
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-150"></div>
              </div>
            </div>
            <p className="text-[var(--foreground)] text-center mb-3">
              {loadingMessage || messages.common?.loading || 'Loading...'}
              {isExporting && (messages.loading?.preparingDownload || ' Please wait while we prepare your download...')}
            </p>

            {/* Long process warning message */}
            {wikiStructure && (
              <div className="w-full max-w-md mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-md">
                <p className="text-xs text-[var(--foreground)] text-center">
                  {messages.wikiProgress?.longProcessWarning || 
                    'This process may take several minutes depending on the size of your codebase. Feel free to browse other projects and come back later.'}
                </p>
              </div>
            )}

            {/* Progress bar for page generation */}
            {wikiStructure && (
              <div className="w-full max-w-md mt-3">
                <div className="bg-[var(--background)]/50 rounded-full h-2 mb-3 overflow-hidden border border-[var(--border-color)]">
                  <div
                    className="bg-[var(--accent-primary)] h-2 rounded-full transition-all duration-300 ease-in-out"
                    style={{
                      width: `${Math.max(5, 100 * (wikiStructure.pages.length - pagesInProgress.size) / wikiStructure.pages.length)}%`
                    }}
                  />
                </div>
                <p className="text-xs text-[var(--muted)] text-center">
                  {language === 'ja'
                    ? `${wikiStructure.pages.length}ページ中${wikiStructure.pages.length - pagesInProgress.size}ページ完了`
                    : messages.repoPage?.pagesCompleted
                        ? messages.repoPage.pagesCompleted
                            .replace('{completed}', (wikiStructure.pages.length - pagesInProgress.size).toString())
                            .replace('{total}', wikiStructure.pages.length.toString())
                        : `${wikiStructure.pages.length - pagesInProgress.size} of ${wikiStructure.pages.length} pages completed`}
                </p>

                {/* Minimize button */}
                <div className="mt-4 flex justify-center">
                  <button
                    onClick={() => {
                      // Just minimize - hides the loading overlay, shows wiki content underneath
                      // Generation continues in background (component stays mounted)
                      // No navigation needed - we're already on the wiki page
                      minimizeProgress();
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 transition-colors border border-[var(--border-color)] text-sm"
                  >
                    <FaMinusSquare className="text-sm" />
                    {messages.wikiProgress?.minimize || 'Minimize'}
                    <span className="text-xs text-[var(--muted)] ml-1">
                      ({messages.wikiProgress?.minimizeDescription || 'Run in background'})
                    </span>
                  </button>
                </div>

                {/* Show list of in-progress pages */}
                {pagesInProgress.size > 0 && (
                  <div className="mt-4 text-xs">
                    <p className="text-[var(--muted)] mb-2">
                      {messages.repoPage?.currentlyProcessing || 'Currently processing:'}
                    </p>
                    <ul className="text-[var(--foreground)] space-y-1">
                      {Array.from(pagesInProgress).slice(0, 3).map(pageId => {
                        const page = wikiStructure.pages.find(p => p.id === pageId);
                        return page ? <li key={pageId} className="truncate border-l-2 border-[var(--accent-primary)]/30 pl-2">{page.title}</li> : null;
                      })}
                      {pagesInProgress.size > 3 && (
                        <li className="text-[var(--muted)]">
                          {language === 'ja'
                            ? `...他に${pagesInProgress.size - 3}ページ`
                            : messages.repoPage?.andMorePages
                                ? messages.repoPage.andMorePages.replace('{count}', (pagesInProgress.size - 3).toString())
                                : `...and ${pagesInProgress.size - 3} more`}
                        </li>
                      )}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : error ? (
          <div className="bg-[var(--highlight)]/5 border border-[var(--highlight)]/30 rounded-lg p-5 mb-4 shadow-sm">
            <div className="flex items-center text-[var(--highlight)] mb-3">
              <FaExclamationTriangle className="mr-2" />
              <span className="font-semibold">{messages.repoPage?.errorTitle || messages.common?.error || 'Error'}</span>
            </div>
            <p className="text-[var(--foreground)] text-sm mb-3">{error}</p>
            <p className="text-[var(--muted)] text-xs">
              {embeddingError ? (
                messages.repoPage?.embeddingErrorDefault || 'This error is related to the document embedding system used for analyzing your repository. Please verify your embedding model configuration, API keys, and try again. If the issue persists, consider switching to a different embedding provider in the model settings.'
              ) : (
                messages.repoPage?.errorMessageDefault || 'Please check that your repository exists and is public. Valid formats are "owner/repo", "https://github.com/owner/repo", "https://gitlab.com/owner/repo", "https://bitbucket.org/owner/repo", or local folder paths like "C:\\path\\to\\folder" or "/path/to/folder".'
              )}
            </p>
            <div className="mt-5 flex gap-3">
              {/* Show Settings button for Azure DevOps token error */}
              {effectiveRepoInfo.type === 'azuredevops' && error.includes('Personal Access Token') && (
                <button
                  onClick={() => setIsModelSelectionModalOpen(true)}
                  className="btn-azure px-5 py-2 inline-flex items-center gap-1.5"
                >
                  <FaCog className="text-sm" />
                  {messages.repoPage?.settings || 'Settings'}
                </button>
              )}
              <Link
                href="/"
                className="btn-azure px-5 py-2 inline-flex items-center gap-1.5"
              >
                <FaHome className="text-sm" />
                {messages.repoPage?.backToHome || 'Back to Home'}
              </Link>
            </div>
          </div>
        ) : wikiStructure ? (
          <div className="h-full flex flex-col lg:flex-row gap-4 w-full overflow-hidden">
            {/* Partial Cache Info Banner */}
            {partialCacheMessage && (
              <div className="absolute top-0 left-0 right-0 z-10 bg-amber-500/10 border-b border-amber-500/30 px-4 py-3">
                <div className="flex items-center justify-between max-w-7xl mx-auto">
                  <div className="flex items-center text-amber-600 dark:text-amber-400 text-sm">
                    <FaExclamationTriangle className="mr-2 flex-shrink-0" />
                    <span>{partialCacheMessage}</span>
                  </div>
                  <button
                    onClick={() => setIsModelSelectionModalOpen(true)}
                    className="ml-4 px-3 py-1 text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-700 dark:text-amber-300 rounded-md border border-amber-500/30 transition-colors flex-shrink-0"
                  >
                    <FaCog className="inline mr-1" />
                    Settings
                  </button>
                </div>
              </div>
            )}
            {/* Wiki Section (Left side - 2/3 on large screens) */}
            <div className={`h-full flex flex-col lg:flex-row gap-4 overflow-hidden bg-[var(--card-bg)] rounded shadow-custom card-azure transition-all duration-300 ${isChatPanelCollapsed ? 'w-full' : 'w-full lg:w-2/3'} ${partialCacheMessage ? 'mt-12' : ''}`}>
              {/* Wiki Navigation */}
              <div className="h-full w-full lg:w-[280px] xl:w-[320px] flex-shrink-0 bg-[var(--background)]/50 rounded-lg rounded-r-none p-5 border-b lg:border-b-0 lg:border-r border-[var(--border-color)] overflow-y-auto">
                <h3 className="text-lg font-semibold text-[var(--foreground)] mb-3">{wikiStructure.title}</h3>
                <p className="text-[var(--muted)] text-sm mb-5 leading-relaxed">{wikiStructure.description}</p>

                {/* Display repository info */}
                <div className="text-xs text-[var(--muted)] mb-5 flex items-center">
                  {effectiveRepoInfo.type === 'local' ? (
                    <div className="flex items-center">
                      <FaFolder className="mr-2" />
                      <span className="break-all">{effectiveRepoInfo.localPath}</span>
                    </div>
                  ) : (
                    <>
                      {effectiveRepoInfo.type === 'github' ? (
                        <FaGithub className="mr-2" />
                      ) : effectiveRepoInfo.type === 'gitlab' ? (
                        <FaGitlab className="mr-2" />
                      ) : (
                        <FaBitbucket className="mr-2" />
                      )}
                      <a
                        href={effectiveRepoInfo.repoUrl ?? ''}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-[var(--accent-primary)] transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)]"
                      >
                        {effectiveRepoInfo.owner}/{effectiveRepoInfo.repo}
                      </a>
                    </>
                  )}
                </div>

                {/* Branch Indicator - displayed for all wikis */}
                <div className="mb-3 flex items-center text-xs text-[var(--muted)]">
                  <span className="mr-2">Branch:</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                    {effectiveRepoInfo.branch || defaultBranch || 'default'}
                  </span>
                </div>

                {/* Wiki Type Indicator */}
                <div className="mb-3 flex items-center text-xs text-[var(--muted)]">
                  <span className="mr-2">Wiki Type:</span>
                  <span className={`px-2 py-0.5 rounded-full ${isComprehensiveView
                    ? 'bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30'
                    : 'bg-[var(--background)] text-[var(--foreground)] border border-[var(--border-color)]'}`}>
                    {isComprehensiveView
                      ? (messages.form?.comprehensive || 'Comprehensive')
                      : (messages.form?.concise || 'Concise')}
                  </span>
                </div>

                {/* Refresh Wiki button */}
                <div className="mb-5">
                  <button
                    onClick={() => setIsModelSelectionModalOpen(true)}
                    disabled={isLoading}
                    className="flex items-center w-full text-xs px-3 py-2 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-colors hover:cursor-pointer"
                  >
                    <FaSync className={`mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                    {messages.repoPage?.refreshWiki || 'Refresh Wiki'}
                  </button>
                </div>

                {/* Export buttons */}
                {Object.keys(generatedPages).length > 0 && (
                  <div className="mb-5">
                    <h4 className="text-sm font-semibold text-[var(--foreground)] mb-3">
                      {messages.repoPage?.exportWiki || 'Export Wiki'}
                    </h4>
                    <div className="flex flex-col gap-2">
                      <button
                        onClick={() => exportWiki('markdown')}
                        disabled={isExporting}
                        className="btn-azure flex items-center text-xs px-3 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <FaDownload className="mr-2" />
                        {messages.repoPage?.exportAsMarkdown || 'Export as Markdown'}
                      </button>
                      <button
                        onClick={() => exportWiki('json')}
                        disabled={isExporting}
                        className="flex items-center text-xs px-3 py-2 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-colors"
                      >
                        <FaFileExport className="mr-2" />
                        {messages.repoPage?.exportAsJson || 'Export as JSON'}
                      </button>
                    </div>
                    {exportError && (
                      <div className="mt-2 text-xs text-[var(--highlight)]">
                        {exportError}
                      </div>
                    )}
                  </div>
                )}

                <h4 className="text-md font-semibold text-[var(--foreground)] mb-3">
                  {messages.repoPage?.pages || 'Pages'}
                </h4>
                <WikiTreeView
                  wikiStructure={wikiStructure}
                  currentPageId={currentPageId}
                  onPageSelect={handlePageSelect}
                  messages={messages.repoPage}
                />
              </div>

              {/* Wiki Content */}
              <div id="wiki-content" className="w-full flex-grow p-6 lg:p-8 overflow-y-auto">
                {currentPageId && generatedPages[currentPageId] ? (
                  <div className="max-w-[900px] xl:max-w-[1000px] mx-auto">
                    <h3 className="text-xl font-semibold text-[var(--foreground)] mb-4 break-words">
                      {generatedPages[currentPageId].title}
                    </h3>



                    <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
                      <Markdown
                        content={processCitations(
                          generatedPages[currentPageId].content, 
                          effectiveRepoInfo, 
                          detectCurrentBranch(effectiveRepoInfo, 'master') || 'master'
                        )}
                      />
                    </div>

                    {generatedPages[currentPageId].relatedPages.length > 0 && (
                      <div className="mt-8 pt-4 border-t border-[var(--border-color)]">
                        <h4 className="text-sm font-semibold text-[var(--muted)] mb-3">
                          {messages.repoPage?.relatedPages || 'Related Pages:'}
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {generatedPages[currentPageId].relatedPages.map(relatedId => {
                            const relatedPage = wikiStructure.pages.find(p => p.id === relatedId);
                            return relatedPage ? (
                              <button
                                key={relatedId}
                                className="bg-[var(--accent-primary)]/10 hover:bg-[var(--accent-primary)]/20 text-xs text-[var(--accent-primary)] px-3 py-1.5 rounded-md transition-colors truncate max-w-full border border-[var(--accent-primary)]/20"
                                onClick={() => handlePageSelect(relatedId)}
                              >
                                {relatedPage.title}
                              </button>
                            ) : null;
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center p-8 text-[var(--muted)] h-full">
                    <div className="relative mb-4">
                      <div className="absolute -inset-2 bg-[var(--accent-primary)]/5 rounded-full blur-md"></div>
                      <FaBookOpen className="text-4xl relative z-10" />
                    </div>
                    <p className="">
                      {messages.repoPage?.selectPagePrompt || 'Select a page from the navigation to view its content'}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Chat Panel (Right side - 1/3 on large screens) */}
            <div className={`h-full flex-shrink-0 transition-all duration-300 ${isChatPanelCollapsed ? 'hidden lg:block lg:w-12' : 'w-full lg:w-1/3 min-w-[300px]'}`}>
              {/* Collapsed state - just show expand button */}
              {isChatPanelCollapsed && (
                <div className="hidden lg:flex h-full items-start pt-4">
                  <button
                    onClick={() => setIsChatPanelCollapsed(false)}
                    className="w-10 h-10 rounded-full bg-[var(--accent-primary)] text-white shadow-lg flex items-center justify-center hover:bg-[var(--accent-primary)]/90 transition-all"
                    aria-label={messages.ask?.title || 'Ask about this repository'}
                    title={messages.ask?.title || 'Ask about this repository'}
                  >
                    <FaComments className="text-lg" />
                  </button>
                </div>
              )}
              
              {/* Chat panel - always rendered but hidden when collapsed to preserve state */}
              <div className={`h-full bg-[var(--card-bg)] rounded shadow-custom card-azure flex flex-col overflow-hidden ${isChatPanelCollapsed ? 'hidden' : ''}`}>
                {/* Chat Header */}
                <div className="flex items-center justify-between p-3 border-b border-[var(--border-color)] bg-[var(--background)]/50">
                  <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                    <FaComments className="text-[var(--accent-primary)]" />
                    {messages.ask?.title || 'Ask about this repository'}
                  </h3>
                  <button
                    onClick={() => setIsChatPanelCollapsed(true)}
                    className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors p-1.5 rounded-md hover:bg-[var(--background)]"
                    aria-label="Collapse chat"
                    title="Collapse chat"
                  >
                    <FaTimes className="text-sm" />
                  </button>
                </div>
                {/* Chat Content */}
                <div className="flex-1 overflow-y-auto p-4">
                  <Ask
                    repoInfo={effectiveRepoInfo}
                    provider={selectedProviderState}
                    model={selectedModelState}
                    isCustomModel={isCustomSelectedModelState}
                    customModel={customSelectedModelState}
                    language={language}
                    onRef={(ref) => (askComponentRef.current = ref)}
                  />
                </div>
              </div>
            </div>

            {/* Mobile Chat Toggle Button - only shown on small screens when chat is collapsed */}
            {isChatPanelCollapsed && (
              <button
                onClick={() => setIsChatPanelCollapsed(false)}
                className="lg:hidden fixed bottom-6 right-6 w-14 h-14 rounded-full bg-[var(--accent-primary)] text-white shadow-lg flex items-center justify-center hover:bg-[var(--accent-primary)]/90 transition-all z-50"
                aria-label={messages.ask?.title || 'Ask about this repository'}
              >
                <FaComments className="text-xl" />
              </button>
            )}
          </div>
        ) : null}
      </main>

      {/* Footer - only shown when chat panel is collapsed or on smaller screens */}
      <footer className={`max-w-[90%] xl:max-w-[1400px] mx-auto mt-8 flex flex-col gap-4 w-full ${!isChatPanelCollapsed && wikiStructure ? 'hidden lg:hidden' : ''}`}>
        <div className="flex justify-between items-center gap-4 text-[var(--muted)] text-sm h-fit w-full bg-[var(--card-bg)] rounded-lg p-3 shadow-sm border border-[var(--border-color)]">
          <p className="shrink-0 text-xs opacity-70 whitespace-nowrap">
            {messages.footer?.brand || '© Microsoft | Azure'}
          </p>
          <p className="flex-1 text-center">
            {messages.footer?.copyright || 'DaP CN | Orcas CodeWiki - AI-powered documentation for repositories on Azure DevOps'}
          </p>
          <div className="shrink-0">
            <ThemeToggle />
          </div>
        </div>
      </footer>

      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProviderState}
        setProvider={setSelectedProviderState}
        model={selectedModelState}
        setModel={setSelectedModelState}
        isCustomModel={isCustomSelectedModelState}
        setIsCustomModel={setIsCustomSelectedModelState}
        customModel={customSelectedModelState}
        setCustomModel={setCustomSelectedModelState}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={true}
        excludedDirs={modelExcludedDirs}
        setExcludedDirs={setModelExcludedDirs}
        excludedFiles={modelExcludedFiles}
        setExcludedFiles={setModelExcludedFiles}
        includedDirs={modelIncludedDirs}
        setIncludedDirs={setModelIncludedDirs}
        includedFiles={modelIncludedFiles}
        setIncludedFiles={setModelIncludedFiles}
        onApply={confirmRefresh}
        showWikiType={true}
        showTokenInput={effectiveRepoInfo.type !== 'local'} // Always show token input for refresh (for git pull)
        repositoryType={effectiveRepoInfo.type as 'github' | 'gitlab' | 'bitbucket' | 'azuredevops'}
        authRequired={authRequired}
        authCode={authCode}
        setAuthCode={setAuthCode}
        isAuthLoading={isAuthLoading}
      />
    </div>
  );
}
