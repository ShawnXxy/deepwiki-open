'use client';

import React, { useState } from 'react';
import { FaChevronRight, FaChevronDown } from 'react-icons/fa';
import logger from '../utils/logger';

// Import interfaces from the page component
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

interface WikiSection {
  id: string;
  title: string;
  pages: string[];
  subsections?: WikiSection[] | string[];
}

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];
}

interface WikiTreeViewProps {
  wikiStructure: WikiStructure;
  currentPageId: string | undefined;
  onPageSelect: (pageId: string) => void;
  messages?: {
    pages?: string;
    [key: string]: string | undefined;
  };
}

const WikiTreeView: React.FC<WikiTreeViewProps> = ({
  wikiStructure,
  currentPageId,
  onPageSelect,
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(wikiStructure.rootSections)
  );

  const toggleSection = (sectionId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  const renderSection = (sectionId: string, level = 0) => {
    const section = wikiStructure.sections.find(s => s.id === sectionId);
    if (!section) return null;

    const isExpanded = expandedSections.has(sectionId);
    // The overview page has the same ID as the section — show it
    // via the section header instead of duplicating as a child.
    const hasOverviewPage = wikiStructure.pages.some(p => p.id === sectionId);
    // Also skip pages that match a subsection ID — they'll render
    // as the subsection header, not as a plain child page.
    const subsectionIds = new Set(
      (section.subsections || []).map(s => typeof s === 'string' ? s : s.id)
    );
    const childPages = section.pages.filter(
      pid => pid !== sectionId && !subsectionIds.has(pid)
    );
    const isOverviewSelected = currentPageId === sectionId;

    return (
      <div key={sectionId} className="mb-2">
        <div className="flex items-center">
          <button
            className="flex-shrink-0 w-6 flex items-center justify-center rounded hover:bg-[var(--background)]/70 transition-colors"
            onClick={(e) => toggleSection(sectionId, e)}
          >
            {isExpanded ? (
              <FaChevronDown className="text-xs text-[var(--foreground)]" />
            ) : (
              <FaChevronRight className="text-xs text-[var(--foreground)]" />
            )}
          </button>
          <button
            className={`flex-1 text-left px-1 py-1.5 rounded-md text-sm font-medium transition-colors ${
              isOverviewSelected
                ? 'text-[var(--accent-primary)]'
                : 'text-[var(--foreground)] hover:bg-[var(--background)]/70'
            } ${level === 0 ? 'bg-[var(--background)]/50' : ''}`}
            onClick={() => {
              if (hasOverviewPage) {
                onPageSelect(sectionId);
              } else {
                toggleSection(sectionId, { stopPropagation: () => {} } as React.MouseEvent);
              }
            }}
          >
            <span className="truncate"><span className="text-[var(--muted)] mr-1.5 font-normal">{sectionId}</span>{section.title}</span>
          </button>
        </div>

        {isExpanded && (
          <div className="ml-4 mt-1 space-y-1 pl-2 border-l border-[var(--border-color)]/30">
            {/* Render child pages (skip the overview page — it's the section header) */}
            {childPages.map(pageId => {
              const page = wikiStructure.pages.find(p => p.id === pageId);
              logger.debug(`WikiTreeView: Looking for pageId "${pageId}"`, { found: page ? page.id : 'NOT FOUND' });
              if (!page) return null;

              return (
                <button
                  key={pageId}
                  className={`w-full text-left px-3 py-1.5 rounded-md text-sm transition-colors ${
                    currentPageId === pageId
                      ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30'
                      : 'text-[var(--foreground)] hover:bg-[var(--background)] border border-transparent'
                  }`}
                  onClick={() => onPageSelect(pageId)}
                >
                  <div className="flex items-center">
                    <span className="text-[var(--muted)] mr-1.5 text-xs flex-shrink-0">{pageId}</span>
                    <span className="truncate">{page.title}</span>
                  </div>
                </button>
              );
            })}

            {/* Render subsections recursively */}
            {section.subsections?.map(sub => {
              if (typeof sub === 'string') {
                // Legacy: subsection is a string ID
                return renderSection(sub, level + 1);
              } else {
                // New: subsection is a WikiSection object
                return renderSectionObj(sub, level + 1);
              }
            })}
          </div>
        )}
      </div>
    );
  };

  // Render a WikiSection object directly (for nested subsections)
  const renderSectionObj = (section: WikiSection, level: number) => {
    const isExpanded = expandedSections.has(section.id);
    const hasOverviewPage = wikiStructure.pages.some(p => p.id === section.id);
    const subsectionIds = new Set(
      (section.subsections || []).map(s => typeof s === 'string' ? s : s.id)
    );
    const childPages = section.pages.filter(
      pid => pid !== section.id && !subsectionIds.has(pid)
    );
    const isOverviewSelected = currentPageId === section.id;
    const hasChildren = childPages.length > 0 || (section.subsections && section.subsections.length > 0);

    return (
      <div key={section.id} className="mb-1">
        <div className="flex items-center">
          {hasChildren ? (
            <button
              className="flex-shrink-0 w-6 flex items-center justify-center rounded hover:bg-[var(--background)]/70 transition-colors"
              onClick={(e) => toggleSection(section.id, e)}
            >
              {isExpanded ? <FaChevronDown className="text-xs" /> : <FaChevronRight className="text-xs" />}
            </button>
          ) : <span className="w-6 flex-shrink-0" />}
          <button
            className={`flex-1 text-left px-1 py-1 rounded-md text-sm font-medium transition-colors ${
              isOverviewSelected
                ? 'text-[var(--accent-primary)]'
                : 'text-[var(--foreground)] hover:bg-[var(--background)]/70'
            }`}
            onClick={() => {
              if (hasOverviewPage) {
                onPageSelect(section.id);
              } else if (hasChildren) {
                toggleSection(section.id, { stopPropagation: () => {} } as React.MouseEvent);
              }
            }}
          >
            <span className="truncate"><span className="text-[var(--muted)] mr-1.5 font-normal">{section.id}</span>{section.title}</span>
          </button>
        </div>

        {isExpanded && (
          <div className={`ml-4 mt-1 space-y-1 pl-2 border-l border-[var(--border-color)]/30`}>
            {childPages.map(pageId => {
              const page = wikiStructure.pages.find(p => p.id === pageId);
              if (!page) return null;
              return (
                <button
                  key={pageId}
                  className={`w-full text-left px-3 py-1.5 rounded-md text-sm transition-colors ${
                    currentPageId === pageId
                      ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30'
                      : 'text-[var(--foreground)] hover:bg-[var(--background)] border border-transparent'
                  }`}
                  onClick={() => onPageSelect(pageId)}
                >
                  <span className="truncate"><span className="text-[var(--muted)] mr-1.5 text-xs">{pageId}</span>{page.title}</span>
                </button>
              );
            })}
            {section.subsections?.map(sub =>
              typeof sub === 'string'
                ? renderSection(sub, level + 1)
                : renderSectionObj(sub, level + 1)
            )}
          </div>
        )}
      </div>
    );
  };

  // If there are no sections defined yet, or if sections/rootSections are empty arrays, fall back to the flat list view
  if (!wikiStructure.sections || wikiStructure.sections.length === 0 || !wikiStructure.rootSections || wikiStructure.rootSections.length === 0) {
    logger.info("WikiTreeView: Falling back to flat list view due to missing or empty sections/rootSections");
    return (
      <ul className="space-y-2">
        {wikiStructure.pages.map(page => (
          <li key={page.id}>
            <button
              className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                currentPageId === page.id
                  ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30'
                  : 'text-[var(--foreground)] hover:bg-[var(--background)] border border-transparent'
              }`}
              onClick={() => onPageSelect(page.id)}
            >
              <div className="flex items-center">
                <div
                  className={`w-2 h-2 rounded-full mr-2 flex-shrink-0 ${
                    page.importance === 'high'
                      ? 'bg-[#0078d4]'
                      : page.importance === 'medium'
                      ? 'bg-[#50e6ff]'
                      : 'bg-[#a0aec0]'
                  }`}
                ></div>
                <span className="truncate">{page.title}</span>
              </div>
            </button>
          </li>
        ))}
      </ul>
    );
  }

  // Log information about the sections for debugging
  logger.debug("WikiTreeView: Rendering tree view", { 
    sectionsCount: wikiStructure.sections.length,
    rootSectionsCount: wikiStructure.rootSections.length,
    pagesCount: wikiStructure.pages.length,
    pageIds: wikiStructure.pages.map(p => p.id)
  });

  return (
    <div className="space-y-1">
      {wikiStructure.rootSections.map(sectionId => {
        const section = wikiStructure.sections.find(s => s.id === sectionId);
        if (!section) {
          logger.warn(`WikiTreeView: Could not find section with id ${sectionId}`);
          return null;
        }
        return renderSection(sectionId);
      })}
    </div>
  );
};

export default WikiTreeView;