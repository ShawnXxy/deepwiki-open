'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { FaHome, FaSearch, FaCode, FaArrowRight } from 'react-icons/fa';
import { useCodeTrace } from '@/hooks/useCodeTrace';
import type { CodeTraceSection, CodeReference, SourceChunk } from '@/types/codetrace';
import ThemeToggle from '@/components/theme-toggle';

// ─── Code Reference Box ──────────────────────────────────────
function CodeRefBox({
  ref: codeRef,
  isActive,
  onClick,
}: {
  ref: CodeReference;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border p-2.5 transition-all text-xs font-mono ${
        isActive
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 ring-1 ring-blue-300'
          : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 hover:border-blue-300'
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="font-semibold text-blue-600 dark:text-blue-400">
          {codeRef.refId && <span className="mr-1.5 text-[10px] bg-blue-100 dark:bg-blue-900 px-1 py-0.5 rounded">{codeRef.refId}</span>}
          {codeRef.filePath.split('/').pop()}:{codeRef.startLine}
        </span>
        <span className="text-[10px] text-gray-400">{codeRef.filePath}</span>
      </div>
      {codeRef.annotation && (
        <div className="text-[11px] text-gray-600 dark:text-gray-300 font-sans">{codeRef.annotation}</div>
      )}
      {codeRef.snippet && (
        <pre className="mt-1.5 text-[10px] text-gray-500 dark:text-gray-400 overflow-x-auto whitespace-pre-wrap max-h-20 leading-relaxed">{codeRef.snippet}</pre>
      )}
    </button>
  );
}

// ─── Trace Section ───────────────────────────────────────────
function TraceSectionCard({
  section,
  activeRef,
  onRefClick,
}: {
  section: CodeTraceSection;
  activeRef: string | null;
  onRefClick: (ref: CodeReference) => void;
}) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
      {/* Section header */}
      <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/50 px-2 py-0.5 rounded-full">
            {section.id}
          </span>
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">{section.title}</h3>
        </div>
        <span className="text-[10px] text-gray-400 mt-1 inline-block">AI generated guide</span>
      </div>

      <div className="p-4 space-y-3">
        {/* Motivation */}
        {section.motivation && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">Motivation</div>
            <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed">{section.motivation}</p>
          </div>
        )}

        {/* Code references */}
        {section.codeRefs.length > 0 && (
          <div className="space-y-2">
            {section.codeRefs.map((ref, i) => (
              <CodeRefBox
                key={ref.refId || i}
                ref={ref}
                isActive={activeRef === `${section.id}-${ref.refId || i}`}
                onClick={() => onRefClick(ref)}
              />
            ))}
          </div>
        )}

        {/* Details */}
        {section.details && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">Details</div>
            <div className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">{section.details}</div>
          </div>
        )}
      </div>

      {/* Connection arrow */}
      {section.connections.length > 0 && (
        <div className="flex justify-center py-2 text-gray-300 dark:text-gray-600">
          <FaArrowRight className="text-xs rotate-90" />
        </div>
      )}
    </div>
  );
}

// ─── Source File Viewer ──────────────────────────────────────
function SourceFileViewer({
  files,
  sourceContents,
  activeFile,
  activeLine,
  onFileSelect,
}: {
  files: string[];
  sourceContents: Record<string, SourceChunk[]>;
  activeFile: string | null;
  activeLine: number | null;
  onFileSelect: (file: string) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll to highlighted line when it changes
  useEffect(() => {
    if (activeLine && scrollRef.current) {
      const el = scrollRef.current.querySelector(`[data-line="${activeLine}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [activeLine, activeFile]);

  const chunks = activeFile ? (sourceContents[activeFile] || []) : [];

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
      {/* File tabs */}
      <div className="flex overflow-x-auto border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 px-1 py-1 gap-0.5 flex-shrink-0">
        {files.map(f => (
          <button
            key={f}
            onClick={() => onFileSelect(f)}
            className={`px-2.5 py-1.5 text-[11px] rounded whitespace-nowrap transition-colors flex items-center gap-1 ${
              activeFile === f
                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 font-medium shadow-sm'
                : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            <FaCode className="text-[9px]" />
            {f.split('/').pop()}
          </button>
        ))}
      </div>

      {/* File path bar */}
      {activeFile && (
        <div className="px-3 py-1.5 border-b border-gray-100 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-800/30 flex-shrink-0">
          <span className="text-[10px] text-gray-400 font-mono">{activeFile}</span>
        </div>
      )}

      {/* Source code content */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {activeFile && chunks.length > 0 ? (
          <div className="font-mono text-[11px] leading-[1.6]">
            {chunks.map((chunk, ci) => (
              <div key={ci}>
                {/* Chunk separator */}
                {ci > 0 && (
                  <div className="border-t border-dashed border-gray-200 dark:border-gray-700 my-1 mx-3" />
                )}
                {/* Line numbers + code */}
                {chunk.content.split('\n').map((line, li) => {
                  const lineNum = chunk.startLine + li;
                  const isHighlighted = activeLine !== null &&
                    lineNum >= (activeLine) &&
                    lineNum <= (activeLine + 5);
                  return (
                    <div
                      key={`${ci}-${li}`}
                      data-line={lineNum}
                      className={`flex ${
                        isHighlighted
                          ? 'bg-yellow-100 dark:bg-yellow-900/30 border-l-2 border-yellow-400'
                          : 'border-l-2 border-transparent hover:bg-gray-50 dark:hover:bg-gray-800/50'
                      }`}
                    >
                      <span className="w-12 text-right pr-3 text-gray-400 dark:text-gray-600 select-none flex-shrink-0 text-[10px]">
                        {lineNum}
                      </span>
                      <pre className="flex-1 pr-3 whitespace-pre-wrap break-all text-gray-700 dark:text-gray-300">
                        {line || ' '}
                      </pre>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        ) : activeFile ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-xs p-4">
            No source code chunks available for this file
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 text-xs p-4">
            <FaCode className="text-2xl mb-2 opacity-30" />
            <p>Select a source file tab above</p>
            <p className="text-[10px] mt-1">or click a code reference on the left</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────
export default function CodeTracePage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const owner = params.owner as string;
  const repo = params.repo as string;
  const initialQuery = searchParams.get('q') || '';
  const repoType = searchParams.get('repo_type') || searchParams.get('type') || 'azuredevops';
  const branch = searchParams.get('branch') || null;
  const repoUrl = searchParams.get('repo_url') ? decodeURIComponent(searchParams.get('repo_url')!) : '';

  const [question, setQuestion] = useState(initialQuery);
  const [activeRef, setActiveRef] = useState<string | null>(null);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { data, loading, error, generate } = useCodeTrace();

  // Auto-select first file when trace data loads
  useEffect(() => {
    if (data && !activeFile) {
      const files = Object.keys(data.sourceContents || {});
      if (files.length > 0) {
        setActiveFile(files[0]);
      }
    }
  }, [data, activeFile]);

  // Auto-generate on load if query present
  useEffect(() => {
    if (initialQuery && repoUrl) {
      generate(initialQuery, repoUrl, repoType, branch);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;
    if (repoUrl) {
      generate(question, repoUrl, repoType, branch);
    }
  };

  const handleRefClick = (ref: CodeReference) => {
    setActiveFile(ref.filePath);
    setActiveLine(ref.startLine);
    setActiveRef(`${ref.refId}`);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm">
        <div className="flex items-center gap-3">
          <Link
            href={`/${owner}/${repo}?repo_url=${encodeURIComponent(repoUrl)}&repo_type=${repoType}${branch ? `&branch=${branch}` : ''}`}
            className="text-blue-600 dark:text-blue-400 hover:text-blue-800 text-sm flex items-center gap-1"
          >
            <FaHome /> ← Back to Wiki
          </Link>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <span className="text-sm text-gray-500">{owner}/{repo}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30 px-2 py-0.5 rounded-full">
            Code Trace
          </span>
          <ThemeToggle />
        </div>
      </header>

      {/* Main 2-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel — Trace sections */}
        <div className="w-[60%] flex flex-col overflow-hidden border-r border-gray-200 dark:border-gray-800">
          {/* Title */}
          {data && (
            <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-800">
              <h1 className="text-base font-bold text-gray-800 dark:text-gray-200">{data.title}</h1>
              <p className="text-[11px] text-gray-400 mt-0.5">
                {data.sections.length} sections · {data.sourceFiles.length} source files
              </p>
            </div>
          )}

          {/* Sections scroll area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {loading && (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <div className="flex gap-1 mb-3">
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" />
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
                <p className="text-sm">Tracing code flow...</p>
                <p className="text-[11px] text-gray-400 mt-1">Analyzing repository and generating trace</p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}

            {data && data.sections.map(section => (
              <TraceSectionCard
                key={section.id}
                section={section}
                activeRef={activeRef}
                onRefClick={handleRefClick}
              />
            ))}

            {!loading && !data && !error && (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <FaSearch className="text-3xl mb-3 opacity-30" />
                <p className="text-sm">Ask a question to trace code flow</p>
                <p className="text-[11px] mt-1">e.g., &quot;How does authentication work?&quot;</p>
              </div>
            )}
          </div>
        </div>

        {/* Right panel — Source files */}
        <div className="w-[40%] flex flex-col overflow-hidden p-3">
          <SourceFileViewer
            files={data?.sourceFiles || Object.keys(data?.sourceContents || {})}
            sourceContents={data?.sourceContents || {}}
            activeFile={activeFile}
            activeLine={activeLine}
            onFileSelect={setActiveFile}
          />
        </div>
      </div>

      {/* Bottom chat bar */}
      <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-3">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask about the code... e.g., How does the cache invalidation work?"
            className="flex-1 px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 placeholder:text-gray-400 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Tracing...' : 'Trace'}
          </button>
        </form>
      </div>
    </div>
  );
}
