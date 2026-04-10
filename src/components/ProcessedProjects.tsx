'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { FaTimes, FaTh, FaList, FaBookOpen, FaFileAlt } from 'react-icons/fa';
import { ProcessedProject } from '@/hooks/useProcessedProjects';

interface ProcessedProjectsProps {
  showHeader?: boolean;
  /** @deprecated No longer used - all projects are shown in a scrollable container */
  maxItems?: number;
  className?: string;
  messages?: Record<string, Record<string, string>>; // Translation messages with proper typing
  /** Pre-fetched projects - if provided, component won't fetch its own data */
  projects?: ProcessedProject[];
  /** Loading state from parent - only used when projects prop is provided */
  isLoading?: boolean;
  /** Callback when a project is deleted - used to update parent state */
  onProjectDeleted?: (projectId: string) => void;
}

export default function ProcessedProjects({ 
  showHeader = true, 
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  maxItems, 
  className = "",
  messages,
  projects: externalProjects,
  isLoading: externalIsLoading,
  onProjectDeleted
}: ProcessedProjectsProps) {
  const [internalProjects, setInternalProjects] = useState<ProcessedProject[]>([]);
  const [internalIsLoading, setInternalIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'card' | 'list'>('card');
  const [wikiTypeFilter] = useState<'all' | 'comprehensive' | 'concise'>('comprehensive');

  // Use external projects if provided, otherwise use internal state
  const projects = externalProjects ?? internalProjects;
  const isLoading = externalProjects !== undefined ? (externalIsLoading ?? false) : internalIsLoading;
  
  // Function to update projects after deletion
  const removeProject = (projectId: string) => {
    if (onProjectDeleted) {
      onProjectDeleted(projectId);
    } else {
      setInternalProjects(prev => prev.filter(p => p.id !== projectId));
    }
  };

  // Default messages fallback
  const defaultMessages = {
    title: 'Processed Wiki Projects',
    searchPlaceholder: 'Search projects by name, owner, or repository...',
    noProjects: 'No projects found in the server cache. The cache might be empty or the server encountered an issue.',
    noSearchResults: 'No projects match your search criteria.',
    processedOn: 'Processed on:',
    loadingProjects: 'Loading projects...',
    errorLoading: 'Error loading projects:',
    backToHome: 'Back to Home'
  };

  const t = (key: string) => {
    if (messages?.projects?.[key]) {
      return messages.projects[key];
    }
    return defaultMessages[key as keyof typeof defaultMessages] || key;
  };

  // Only fetch if external projects are not provided
  useEffect(() => {
    // Skip fetching if projects are provided externally
    if (externalProjects !== undefined) {
      return;
    }

    const fetchProjects = async () => {
      setInternalIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/wiki/projects');
        if (!response.ok) {
          throw new Error(`Failed to fetch projects: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }
        setInternalProjects(data as ProcessedProject[]);
      } catch (e: unknown) {
        console.error("Failed to load projects from API:", e);
        const message = e instanceof Error ? e.message : "An unknown error occurred.";
        setError(message);
        setInternalProjects([]);
      } finally {
        setInternalIsLoading(false);
      }
    };

    fetchProjects();
  }, [externalProjects]);

  // Filter projects based on search query and wiki type
  const filteredProjects = useMemo(() => {
    let filtered = projects;

    // Apply wiki type filter
    if (wikiTypeFilter === 'comprehensive') {
      filtered = filtered.filter(project => project.comprehensive === true);
    } else if (wikiTypeFilter === 'concise') {
      filtered = filtered.filter(project => project.comprehensive === false);
    }

    // Apply search query filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(project => 
        project.name.toLowerCase().includes(query) ||
        project.owner.toLowerCase().includes(query) ||
        project.repo.toLowerCase().includes(query) ||
        project.repo_type.toLowerCase().includes(query)
      );
    }

    // Note: maxItems is now ignored - all filtered projects are shown in a scrollable container
    return filtered;
  }, [projects, searchQuery, wikiTypeFilter]);

  const clearSearch = () => {
    setSearchQuery('');
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleDelete = async (project: ProcessedProject) => {
    const modeLabel = project.comprehensive ? 'comprehensive' : 'concise';
    if (!confirm(`Are you sure you want to delete the ${modeLabel} wiki for ${project.name}?`)) {
      return;
    }
    try {
      const response = await fetch('/api/wiki/projects', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: project.owner,
          repo: project.repo,
          repo_type: project.repo_type,
          language: project.language,
          comprehensive: project.comprehensive,
          branch: project.branch,
        }),
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({ error: response.statusText }));
        throw new Error(errorBody.error || response.statusText);
      }
      removeProject(project.id);
    } catch (e: unknown) {
      console.error('Failed to delete project:', e);
      alert(`Failed to delete project: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
  };

  return (
    <div className={`flex flex-col flex-1 min-h-0 ${className}`}>
      {showHeader && (
        <header className="mb-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-[var(--accent-primary)]">{t('title')}</h1>
            <Link href="/" className="text-[var(--accent-primary)] hover:underline">
              {t('backToHome')}
            </Link>
          </div>
        </header>
      )}

      {/* Search Bar and View Toggle */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        {/* Search Bar */}
        <div className="relative flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t('searchPlaceholder')}
            className="input-azure block w-full pl-4 pr-12 py-3 border-2 border-[var(--border-color)] rounded-2xl bg-[var(--input-bg)] text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:border-[var(--accent-primary)] focus:ring-4 focus:ring-[var(--accent-primary)]/12 transition-all"
          />
          {searchQuery && (
            <button
              onClick={clearSearch}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              <FaTimes className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* View Toggle */}
        <div className="flex items-center bg-[var(--card-bg-solid)] border border-[var(--border-color)] rounded-2xl p-1.5">
          <button
            onClick={() => setViewMode('card')}
            className={`p-2.5 rounded-xl transition-all ${
              viewMode === 'card'
                ? 'text-white shadow-lg'
                : 'text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--accent-primary)]/5'
            }`}
            style={viewMode === 'card' ? { background: 'var(--gradient-primary)' } : undefined}
            title="Card View"
          >
            <FaTh className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-2.5 rounded-xl transition-all ${
              viewMode === 'list'
                ? 'text-white shadow-lg'
                : 'text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--accent-primary)]/5'
            }`}
            style={viewMode === 'list' ? { background: 'var(--gradient-primary)' } : undefined}
            title="List View"
          >
            <FaList className="h-4 w-4" />
          </button>
        </div>
      </div>

      {isLoading && <p className="text-[var(--muted)]">{t('loadingProjects')}</p>}
      {error && <p className="text-[var(--highlight)]">{t('errorLoading')} {error}</p>}

      {!isLoading && !error && filteredProjects.length > 0 && (
        <div className="flex-1 min-h-0 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-[var(--border-color)] scrollbar-track-transparent">
          <div className={viewMode === 'card' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-2'}>
            {filteredProjects.map((project) => (
            viewMode === 'card' ? (
              <div key={project.id} className="relative p-5 border border-[var(--border-color)] rounded-2xl bg-[var(--card-bg)] shadow-custom hover:shadow-elevated transition-all duration-300 hover:scale-[1.02] hover:border-[var(--accent-primary)]/20 group">
                {/* Delete button hidden - users cannot remove existing wikis */}
                <Link
                  href={`/${project.owner}/${project.repo}?type=${project.repo_type}&language=${project.language}&comprehensive=${project.comprehensive}${project.branch ? `&branch=${project.branch}` : ''}`}
                  className="block"
                >
                  <h3 className="text-lg font-semibold text-[var(--link-color)] hover:underline mb-2 line-clamp-2">
                    {project.name}
                  </h3>
                  <div className="flex flex-wrap gap-2 mb-3">
                    <span className="px-2 py-1 text-xs bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] rounded-full border border-[var(--accent-primary)]/20">
                      {project.repo_type}
                    </span>
                    <span className="px-2 py-1 text-xs bg-[var(--background)] text-[var(--muted)] rounded-full border border-[var(--border-color)]">
                      {project.language}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full border ${
                      project.comprehensive
                        ? 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20'
                        : 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20'
                    }`}>
                      {project.comprehensive ? 'comprehensive' : 'concise'}
                    </span>
                    {project.branch && (
                      <span className="px-2 py-1 text-xs bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-full border border-emerald-500/20">
                        {project.branch}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-[var(--muted)]">
                    {t('processedOn')} {new Date(project.submittedAt).toLocaleDateString()}
                  </p>
                </Link>
              </div>
            ) : (
              <div key={project.id} className="relative p-4 border border-[var(--border-color)] rounded-2xl bg-[var(--card-bg)] hover:bg-[var(--accent-primary)]/5 transition-all hover:border-[var(--accent-primary)]/15">
                {/* Delete button hidden - users cannot remove existing wikis */}
                <Link
                  href={`/${project.owner}/${project.repo}?type=${project.repo_type}&language=${project.language}&comprehensive=${project.comprehensive}${project.branch ? `&branch=${project.branch}` : ''}`}
                  className="flex items-center justify-between"
                >
                  <div className="flex-1 min-w-0">
                    <h3 className="text-base font-medium text-[var(--link-color)] hover:underline truncate">
                      {project.name}
                    </h3>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      {t('processedOn')} {new Date(project.submittedAt).toLocaleDateString()} • {project.repo_type} • {project.language} • {project.comprehensive ? 'comprehensive' : 'concise'}{project.branch ? ` • ${project.branch}` : ''}
                    </p>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <span className="px-2 py-1 text-xs bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] rounded border border-[var(--accent-primary)]/20">
                      {project.repo_type}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded border ${
                      project.comprehensive
                        ? 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20'
                        : 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20'
                    }`}>
                      {project.comprehensive ? 'comprehensive' : 'concise'}
                    </span>
                    {project.branch && (
                      <span className="px-2 py-1 text-xs bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded border border-emerald-500/20">
                        {project.branch}
                      </span>
                    )}
                  </div>
                </Link>
              </div>
            )
          ))}
          </div>
        </div>
      )}

      {!isLoading && !error && projects.length > 0 && filteredProjects.length === 0 && searchQuery && (
        <p className="text-[var(--muted)]">{t('noSearchResults')}</p>
      )}

      {!isLoading && !error && projects.length === 0 && (
        <p className="text-[var(--muted)]">{t('noProjects')}</p>
      )}
    </div>
  );
}
