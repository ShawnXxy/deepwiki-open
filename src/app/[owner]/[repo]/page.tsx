'use client';

import Ask from '@/components/Ask';
import Markdown from '@/components/Markdown';
import ThemeToggle from '@/components/theme-toggle';
import WikiTreeView from '@/components/WikiTreeView';
import { useLanguage } from '@/contexts/LanguageContext';
import { RepoInfo } from '@/types/repoinfo';
import { processCitations, generateFileUrl } from '@/utils/citationProcessor';
import { detectCurrentBranch } from '@/utils/branchDetection';
import getRepoUrl from '@/utils/getRepoUrl';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import React, { useCallback, useEffect, useMemo, useRef, useState, lazy, Suspense } from 'react';
import {
  FaBookOpen, FaComments, FaDownload, FaExclamationTriangle,
  FaFileExport, FaFolder, FaHome, FaLink, FaProjectDiagram, FaSearch, FaTimes,
} from 'react-icons/fa';
import { AzureDevOpsIcon } from '@/components/AzureIcon';
import { useCodeMap } from '@/hooks/useCodeMap';

const CodeMap = lazy(() => import('@/components/CodeMap'));

// ─── Types ────────────────────────────────────────────────────
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
}

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];
}

// ─── Styles ───────────────────────────────────────────────────
const wikiStyles = `
  .prose code { @apply bg-[var(--accent-primary)]/5 px-1.5 py-0.5 rounded-lg font-mono text-xs border border-[var(--accent-primary)]/15; }
  .prose pre { @apply bg-[var(--card-bg-solid)] text-[var(--foreground)] rounded-2xl p-5 overflow-x-auto border border-[var(--border-color)] shadow-custom; }
  .prose h1, .prose h2, .prose h3, .prose h4 { @apply font-bold text-[var(--foreground)]; }
  .prose p { @apply text-[var(--foreground)] leading-relaxed; }
  .prose a { @apply text-[var(--accent-primary)] hover:text-[var(--highlight)] transition-colors no-underline border-b border-[var(--accent-primary)]/30 hover:border-[var(--accent-primary)]; }
  .prose blockquote { @apply border-l-4 border-[var(--accent-primary)]/30 bg-[var(--accent-primary)]/5 pl-4 py-2 italic rounded-r-2xl; }
  .prose ul, .prose ol { @apply text-[var(--foreground)]; }
  .prose table { @apply border-collapse border border-[var(--border-color)] rounded-2xl overflow-hidden; }
  .prose th { @apply bg-[var(--accent-primary)]/5 text-[var(--foreground)] p-3 border border-[var(--border-color)] font-semibold; }
  .prose td { @apply p-3 border border-[var(--border-color)]; }
`;

// ─── Component ────────────────────────────────────────────────
export default function RepoWikiPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const owner = params.owner as string;
  const repo = params.repo as string;

  const language = searchParams.get('language') || 'en';
  const branch = searchParams.get('branch') || null;
  const isComprehensiveView = searchParams.get('comprehensive') !== 'false';
  const repoUrl = searchParams.get('repo_url') ? decodeURIComponent(searchParams.get('repo_url') || '') : undefined;
  const initialPageId = searchParams.get('page') || null; // Deep-link to specific page

  // Determine repo type from URL
  const repoType = (() => {
    if (!repoUrl) return searchParams.get('type') || 'azuredevops';
    if (repoUrl.includes('bitbucket')) return 'bitbucket';
    if (repoUrl.includes('gitlab')) return 'gitlab';
    if (repoUrl.includes('github')) return 'github';
    return 'azuredevops';
  })();

  const { messages } = useLanguage();

  const repoInfo = useMemo<RepoInfo>(() => ({
    owner, repo, type: repoType,
    token: null, branch,
    localPath: null, repoUrl: repoUrl || null,
  }), [owner, repo, repoType, repoUrl, branch]);

  // ─── State ──────────────────────────────────────────────────
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wikiStructure, setWikiStructure] = useState<WikiStructure | undefined>();
  const [currentPageId, setCurrentPageId] = useState<string | undefined>();
  const [generatedPages, setGeneratedPages] = useState<Record<string, WikiPage>>({});
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isChatPanelCollapsed, setIsChatPanelCollapsed] = useState(true);
  const askComponentRef = useRef<{ clearConversation: () => void } | null>(null);

  // Cache metadata
  const [commitHash, setCommitHash] = useState<string>('');
  const [indexedAt, setIndexedAt] = useState<string>('');
  const [cachedProvider, setCachedProvider] = useState<string>('');
  const [cachedModel, setCachedModel] = useState<string>('');

  // Effective repo info (may be updated from cache data)
  const [effectiveRepoInfo, setEffectiveRepoInfo] = useState(repoInfo);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Share feedback
  const [shareCopied, setShareCopied] = useState(false);

  // CodeMap tab
  const [activeView, setActiveView] = useState<'wiki' | 'codemap'>('wiki');
  const { data: codeMapData, loading: codeMapLoading, error: codeMapError } = useCodeMap(
    owner, repo, repoType, branch,
  );

  // Scroll to top on page change
  useEffect(() => {
    const el = document.getElementById('wiki-content');
    if (el) el.scrollTo({ top: 0, behavior: 'smooth' });
  }, [currentPageId]);

  // ─── Load Cache ─────────────────────────────────────────────
  const effectRan = useRef(false);

  useEffect(() => {
    if (effectRan.current) return;
    effectRan.current = true;

    const loadCache = async () => {
      try {
        const params = new URLSearchParams({
          owner,
          repo,
          repo_type: repoType,
          language,
          comprehensive: isComprehensiveView.toString(),
        });
        if (branch) params.set('branch', branch);

        const response = await fetch(`/api/wiki_cache?${params.toString()}`);

        if (!response.ok) {
          setError(
            'Wiki not available for this repository. ' +
            'Run the code processor to generate it: ' +
            'python -m backend.processor.code_processor --config=backend/run.json'
          );
          setIsLoading(false);
          return;
        }

        const data = await response.json();

        if (!data.wiki_structure || !data.generated_pages) {
          setError('Invalid wiki cache data (missing structure or pages).');
          setIsLoading(false);
          return;
        }

        // Check completeness — warn but still load available pages
        const totalPages = data.wiki_structure.pages?.length || 0;
        const pagesWithContent = Object.values(data.generated_pages as Record<string, WikiPage>)
          .filter((p) => p.content && p.content !== 'Loading...').length;

        if (data.is_partial || pagesWithContent < totalPages) {
          console.warn(`Wiki partially generated: ${pagesWithContent}/${totalPages} pages`);
          // Don't block — still load whatever pages exist
        }

        // Load data into state
        setWikiStructure(data.wiki_structure);
        setGeneratedPages(data.generated_pages);
        setCommitHash(data.commit_hash || '');
        setIndexedAt(data.indexed_at || '');
        setCachedProvider(data.provider || '');
        setCachedModel(data.model || '');

        // Update repo info from cache (may have branch info, repo URL, etc.)
        if (data.repo) {
          setEffectiveRepoInfo(prev => ({
            ...prev,
            owner: data.repo.owner || prev.owner,
            repo: data.repo.repo || prev.repo,
            type: data.repo.type || prev.type,
            branch: data.repo.branch || prev.branch,
            repoUrl: data.repo.repoUrl || prev.repoUrl,
          }));
        }

        // Select page from deep-link or default to first
        if (data.wiki_structure.pages?.length > 0) {
          const targetPage = initialPageId && data.generated_pages[initialPageId]
            ? initialPageId
            : data.wiki_structure.pages[0].id;
          setCurrentPageId(targetPage);
        }
      } catch (err) {
        console.error('Error loading wiki cache:', err);
        setError('Failed to load wiki data. Is the backend running?');
      } finally {
        setIsLoading(false);
      }
    };

    loadCache();
  }, [owner, repo, repoType, language, branch, isComprehensiveView]);

  // ─── Export ─────────────────────────────────────────────────
  const exportWiki = useCallback(async (format: 'markdown' | 'json') => {
    if (!wikiStructure || Object.keys(generatedPages).length === 0) {
      setExportError('No wiki content to export');
      return;
    }
    try {
      setIsExporting(true);
      setExportError(null);

      const pagesToExport = wikiStructure.pages.map(page => ({
        ...page,
        content: generatedPages[page.id]?.content || 'Content not generated',
      }));

      const repoUrl = getRepoUrl(effectiveRepoInfo);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      let content: string;
      let filename: string;

      if (format === 'markdown') {
        // Build Markdown export client-side
        let md = `# Wiki Documentation for ${repoUrl}\n\n`;
        md += `Generated on: ${new Date().toLocaleString()}\n\n`;
        md += `## Table of Contents\n\n`;
        for (const page of pagesToExport) {
          md += `- [${page.title}](#${page.id})\n`;
        }
        md += '\n';
        for (const page of pagesToExport) {
          md += `<a id='${page.id}'></a>\n\n`;
          md += `## ${page.title}\n\n`;
          md += `${page.content}\n\n---\n\n`;
        }
        content = md;
        filename = `${effectiveRepoInfo.repo}_wiki_${timestamp}.md`;
      } else {
        // Build JSON export client-side
        content = JSON.stringify({
          metadata: { repository: repoUrl, generated_at: new Date().toISOString(), page_count: pagesToExport.length },
          pages: pagesToExport,
        }, null, 2);
        filename = `${effectiveRepoInfo.repo}_wiki_${timestamp}.json`;
      }

      // Download the file
      const blob = new Blob([content], { type: format === 'markdown' ? 'text/markdown' : 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Export error:', err);
      setExportError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  }, [wikiStructure, generatedPages, effectiveRepoInfo]);

  // ─── Search ─────────────────────────────────────────────────
  const searchResults = useMemo(() => {
    if (!searchQuery.trim() || !wikiStructure) return null;
    const q = searchQuery.toLowerCase();
    return wikiStructure.pages.filter(p => {
      if (p.title.toLowerCase().includes(q)) return true;
      const content = generatedPages[p.id]?.content || '';
      return content.toLowerCase().includes(q);
    });
  }, [searchQuery, wikiStructure, generatedPages]);

  // Keyboard shortcut: Ctrl+K to open search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(prev => !prev);
        setTimeout(() => searchInputRef.current?.focus(), 50);
      }
      if (e.key === 'Escape' && isSearchOpen) {
        setIsSearchOpen(false);
        setSearchQuery('');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isSearchOpen]);

  // ─── Share ──────────────────────────────────────────────────
  const handleShare = useCallback(() => {
    const url = new URL(window.location.href);
    if (currentPageId) url.searchParams.set('page', currentPageId);
    navigator.clipboard.writeText(url.toString());
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
  }, [currentPageId]);

  // ─── Render ─────────────────────────────────────────────────
  return (
    <div className="h-screen paper-texture p-4 md:p-6 flex flex-col relative overflow-hidden">
      {/* Decorative floating orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden z-0">
        <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-[var(--accent-primary)]/6 blur-3xl animate-float" />
        <div className="absolute -bottom-32 -right-32 w-80 h-80 rounded-full bg-[var(--highlight)]/5 blur-3xl animate-float" style={{ animationDelay: '3s' }} />
      </div>
      <style>{wikiStyles}</style>

      {/* Header */}
      <header className="max-w-[90%] xl:max-w-[1400px] mx-auto mb-6 h-fit w-full animate-fade-in relative z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center gap-1.5 transition-all px-4 py-2 rounded-xl hover:bg-[var(--accent-primary)]/10 font-medium">
              <FaHome /> {messages.repoPage?.home || 'Home'}
            </Link>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className={`flex-1 mx-auto overflow-hidden relative z-10 ${activeView === 'codemap' ? 'w-full px-4' : wikiStructure && !isChatPanelCollapsed ? 'w-full px-4' : 'max-w-[90%] xl:max-w-[1400px]'}`}>
        {isLoading ? (
          /* Loading state */
          <div className="flex flex-col items-center justify-center p-12 glass-surface rounded-3xl shadow-elevated max-w-2xl mx-auto animate-fade-in">
            <div className="relative mb-6">
              <div className="absolute -inset-4 bg-[var(--accent-primary)]/10 rounded-full blur-md animate-pulse"></div>
              <div className="relative flex items-center justify-center">
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse"></div>
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-75 mx-2"></div>
                <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-150"></div>
              </div>
            </div>
            <p className="text-[var(--foreground)] text-center">{messages.common?.loading || 'Loading...'}</p>
          </div>
        ) : error ? (
          /* Error / Wiki not available */
          <div className="bg-[var(--warning)]/8 border border-[var(--warning)]/25 rounded-2xl p-6 mb-4 shadow-custom max-w-2xl mx-auto animate-fade-in">
            <div className="flex items-center text-[var(--highlight)] mb-3">
              <FaExclamationTriangle className="mr-2" />
              <span className="font-semibold">{messages.common?.error || 'Error'}</span>
            </div>
            <p className="text-[var(--foreground)] text-sm mb-3">{error}</p>
            <div className="mt-5 flex gap-3">
              <Link href="/" className="btn-azure px-6 py-2.5 inline-flex items-center gap-2 rounded-xl">
                <FaHome className="text-sm" />
                {messages.repoPage?.backToHome || 'Back to Home'}
              </Link>
            </div>
          </div>
        ) : wikiStructure ? (
          /* Wiki viewer */
          <div className="h-full flex flex-col w-full overflow-hidden">
            {/* Tab switcher */}
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-[var(--border-color)]">
              <button
                onClick={() => setActiveView('wiki')}
                className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all flex items-center gap-2 ${
                  activeView === 'wiki'
                    ? 'bg-gradient-to-r from-[var(--accent-primary)] to-[var(--highlight)] text-white shadow-lg'
                    : 'text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--accent-primary)]/5 glass-surface'
                }`}
              >
                <FaBookOpen className="text-xs" />
                Wiki
              </button>
              <button
                onClick={() => setActiveView('codemap')}
                className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all flex items-center gap-2 ${
                  activeView === 'codemap'
                    ? 'bg-gradient-to-r from-[var(--accent-primary)] to-[var(--highlight)] text-white shadow-lg'
                    : 'text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--accent-primary)]/5 glass-surface'
                }`}
              >
                <FaProjectDiagram className="text-xs" />
                Code Map
                {codeMapData && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/20 font-bold">
                    {codeMapData.metadata.totalFiles}
                  </span>
                )}
              </button>
            </div>

            {activeView === 'codemap' ? (
              /* Code Map view */
              <div className="w-full glass-surface rounded-3xl shadow-elevated" style={{ height: 'calc(100vh - 180px)' }}>
                {codeMapLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="flex items-center gap-2 text-[var(--muted)]">
                      <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse"></div>
                      <p>Loading code map...</p>
                    </div>
                  </div>
                ) : codeMapError ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-[var(--muted)] p-8">
                      <FaProjectDiagram className="text-4xl mx-auto mb-3 opacity-30" />
                      <p className="text-sm">{codeMapError}</p>
                    </div>
                  </div>
                ) : codeMapData ? (
                  <Suspense fallback={
                    <div className="flex items-center justify-center h-full">
                      <p className="text-[var(--muted)]">Loading visualization...</p>
                    </div>
                  }>
                    <CodeMap
                      data={codeMapData}
                      onNavigateToFile={(filePath) => {
                        // Find a wiki page that references this file
                        const matchingPage = wikiStructure.pages.find(
                          p => generatedPages[p.id]?.filePaths?.some(
                            fp => fp === filePath || filePath.endsWith(fp) || fp.endsWith(filePath)
                          )
                        );
                        if (matchingPage) {
                          setCurrentPageId(matchingPage.id);
                          setActiveView('wiki');
                        }
                      }}
                    />
                  </Suspense>
                ) : null}
              </div>
            ) : (
            <div className="flex-1 flex flex-col lg:flex-row gap-4 overflow-hidden">
            {/* Wiki Section (left 2/3) */}
            <div className={`h-full flex flex-col lg:flex-row gap-0 overflow-hidden glass-surface rounded-3xl shadow-elevated transition-all duration-300 ${isChatPanelCollapsed ? 'w-full' : 'w-full lg:w-2/3'}`}>
              {/* Sidebar navigation */}
              <div className="h-full w-full lg:w-[280px] xl:w-[320px] flex-shrink-0 p-5 border-b lg:border-b-0 lg:border-r border-[var(--border-color)] overflow-y-auto scrollbar-thin" style={{ background: 'var(--gradient-sidebar)' }}>
                <h3 className="text-lg font-semibold text-[var(--foreground)] mb-3">{wikiStructure.title}</h3>
                <p className="text-[var(--muted)] text-sm mb-5 leading-relaxed">{wikiStructure.description}</p>

                {/* Repository link */}
                <div className="text-xs text-[var(--muted)] mb-5 flex items-center">
                  {effectiveRepoInfo.type === 'local' ? (
                    <div className="flex items-center">
                      <FaFolder className="mr-2" />
                      <span className="break-all">{effectiveRepoInfo.localPath}</span>
                    </div>
                  ) : (
                    <>
                      <AzureDevOpsIcon className="mr-2" />
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

                {/* Branch indicator */}
                <div className="mb-3 flex items-center text-xs text-[var(--muted)]">
                  <span className="mr-2">Branch:</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                    {effectiveRepoInfo.branch || 'default'}
                  </span>
                </div>

                {/* Wiki type */}
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
                        className="btn-azure flex items-center text-xs px-4 py-3 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <FaDownload className="mr-2" />
                        {messages.repoPage?.exportAsMarkdown || 'Export as Markdown'}
                      </button>
                      <button
                        onClick={() => exportWiki('json')}
                        disabled={isExporting}
                        className="flex items-center text-xs px-4 py-3 bg-[var(--card-bg-solid)] text-[var(--foreground)] rounded-xl hover:bg-[var(--accent-primary)]/5 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-all"
                      >
                        <FaFileExport className="mr-2" />
                        {messages.repoPage?.exportAsJson || 'Export as JSON'}
                      </button>
                    </div>
                    {exportError && (
                      <div className="mt-2 text-xs text-[var(--highlight)]">{exportError}</div>
                    )}
                  </div>
                )}

                {/* Pages heading + search + indexed-at metadata */}
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-md font-semibold text-[var(--foreground)]">
                    {messages.repoPage?.pages || 'Pages'}
                  </h4>
                  <button
                    onClick={() => { setIsSearchOpen(!isSearchOpen); setTimeout(() => searchInputRef.current?.focus(), 50); }}
                    className="p-1.5 text-[var(--muted)] hover:text-[var(--accent-primary)] transition-colors rounded-lg hover:bg-[var(--accent-primary)]/10"
                    title="Search pages (Ctrl+K)"
                  >
                    <FaSearch className="text-xs" />
                  </button>
                </div>

                {/* Search input */}
                {isSearchOpen && (
                  <div className="mb-3">
                    <input
                      ref={searchInputRef}
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search pages..."
                      className="w-full px-4 py-2.5 text-xs bg-[var(--input-bg)] text-[var(--foreground)] border-2 border-[var(--border-color)] rounded-xl focus:outline-none focus:border-[var(--accent-primary)] focus:ring-4 focus:ring-[var(--accent-primary)]/12 placeholder:text-[var(--muted)] transition-all"
                    />
                    {searchResults && searchResults.length > 0 && (
                      <div className="mt-1 max-h-40 overflow-y-auto">
                        {searchResults.map(page => (
                          <button
                            key={page.id}
                            onClick={() => { setCurrentPageId(page.id); setSearchQuery(''); setIsSearchOpen(false); }}
                            className="w-full text-left px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 rounded-lg truncate transition-colors"
                          >
                            {page.title}
                          </button>
                        ))}
                      </div>
                    )}
                    {searchResults && searchResults.length === 0 && searchQuery.trim() && (
                      <p className="mt-1 text-xs text-[var(--muted)] px-3">No pages found</p>
                    )}
                  </div>
                )}

                {indexedAt && (
                  <p className="text-xs text-[var(--muted)] mb-3">
                    Last indexed: {new Date(indexedAt).toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })}
                    {commitHash && ` (${commitHash.slice(0, 7)})`}
                  </p>
                )}

                <WikiTreeView
                  wikiStructure={wikiStructure}
                  currentPageId={currentPageId}
                  onPageSelect={(id) => { if (currentPageId !== id) setCurrentPageId(id); }}
                  messages={messages.repoPage}
                />
              </div>

              {/* Wiki content */}
              <div id="wiki-content" className="w-full flex-grow p-6 lg:p-8 overflow-y-auto">
                {currentPageId && generatedPages[currentPageId] ? (
                  <div className="max-w-[900px] xl:max-w-[1000px] mx-auto">
                    {/* Page title + share button */}
                    <div className="flex items-start justify-between gap-3 mb-4">
                      <h3 className="text-xl font-semibold text-[var(--foreground)] break-words">
                        {generatedPages[currentPageId].title}
                      </h3>
                      <button
                        onClick={handleShare}
                        className="flex-shrink-0 p-2 text-[var(--muted)] hover:text-[var(--accent-primary)] transition-colors rounded-lg hover:bg-[var(--accent-primary)]/10"
                        title="Copy link to this page"
                      >
                        <FaLink className="text-sm" />
                      </button>
                    </div>
                    {shareCopied && (
                      <div className="mb-3 text-xs text-emerald-600 dark:text-emerald-400">Link copied to clipboard</div>
                    )}

                    <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
                      <Markdown
                        content={processCitations(
                          generatedPages[currentPageId].content,
                          effectiveRepoInfo,
                          commitHash || detectCurrentBranch(effectiveRepoInfo, 'master') || 'master',
                        )}
                        onNavigateToPage={(pageId) => setCurrentPageId(pageId)}
                      />
                    </div>

                    {/* Related pages */}
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
                                className="bg-[var(--accent-primary)]/8 hover:bg-[var(--accent-primary)]/15 text-xs text-[var(--accent-primary)] font-medium px-4 py-2 rounded-xl transition-all truncate max-w-full border border-[var(--accent-primary)]/15 hover:shadow-sm hover:scale-[1.02]"
                                onClick={() => setCurrentPageId(relatedId)}
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
                    <p>{messages.repoPage?.selectPagePrompt || 'Select a page from the navigation to view its content'}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Chat panel (right 1/3) */}
            <div className={`h-full flex-shrink-0 transition-all duration-300 ${isChatPanelCollapsed ? 'hidden lg:block lg:w-12' : 'w-full lg:w-1/3 min-w-[300px]'}`}>
              {isChatPanelCollapsed && (
                <div className="hidden lg:flex h-full items-start pt-4">
                  <button
                    onClick={() => setIsChatPanelCollapsed(false)}
                    className="w-10 h-10 rounded-2xl bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] text-white shadow-lg flex items-center justify-center hover:shadow-xl transition-all hover:scale-110"
                    aria-label={messages.ask?.title || 'Ask about this repository'}
                    title={messages.ask?.title || 'Ask about this repository'}
                  >
                    <FaComments className="text-lg" />
                  </button>
                </div>
              )}
              <div className={`h-full glass-surface rounded-3xl shadow-elevated flex flex-col overflow-hidden ${isChatPanelCollapsed ? 'hidden' : ''}`}>
                <div className="flex items-center justify-between p-4 border-b border-[var(--border-color)]">
                  <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                    <FaComments className="text-[var(--accent-primary)]" />
                    {messages.ask?.title || 'Ask about this repository'}
                  </h3>
                  <button
                    onClick={() => setIsChatPanelCollapsed(true)}
                    className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors p-1.5 rounded-lg hover:bg-[var(--background)]"
                    aria-label="Collapse chat"
                  >
                    <FaTimes className="text-sm" />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-4">
                  <Ask
                    repoInfo={effectiveRepoInfo}
                    provider={cachedProvider}
                    model={cachedModel}
                    language={language}
                    isVisible={!isChatPanelCollapsed}
                    onRef={(ref) => (askComponentRef.current = ref)}
                  />
                </div>
              </div>
            </div>

            {/* Mobile chat toggle */}
            {isChatPanelCollapsed && (
              <button
                onClick={() => setIsChatPanelCollapsed(false)}
                className="lg:hidden fixed bottom-6 right-6 w-14 h-14 rounded-2xl bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] text-white shadow-lg flex items-center justify-center hover:shadow-xl transition-all hover:scale-110 z-50"
                aria-label={messages.ask?.title || 'Ask about this repository'}
              >
                <FaComments className="text-xl" />
              </button>
            )}
            </div>
            )}
          </div>
        ) : null}
      </main>

      {/* Footer */}
      <footer className={`max-w-[90%] xl:max-w-[1400px] mx-auto mt-6 flex flex-col gap-4 w-full relative z-10 ${!isChatPanelCollapsed && wikiStructure ? 'hidden lg:hidden' : ''}`}>
        <div className="flex justify-between items-center gap-4 text-[var(--muted)] text-sm h-fit w-full glass-surface rounded-2xl p-3">
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
    </div>
  );
}
