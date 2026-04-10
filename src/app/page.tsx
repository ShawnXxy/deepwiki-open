'use client';

import React from 'react';
import Link from 'next/link';
import ThemeToggle from '@/components/theme-toggle';
import Mermaid from '../components/Mermaid';
import ProcessedProjects from '@/components/ProcessedProjects';
import { AzureDevOpsIcon, MicrosoftLogo } from '@/components/AzureIcon';
import { useProcessedProjects } from '@/hooks/useProcessedProjects';
import { useLanguage } from '@/contexts/LanguageContext';

const DEMO_FLOW_CHART = `graph TD
  A[Code Repository] --> B[DeepWiki]
  B --> C[Architecture Diagrams]
  B --> D[Component Relationships]
  B --> E[Data Flow]
  B --> F[Process Workflows]

  style A fill:#f9d3a9,stroke:#d86c1f
  style B fill:#d4a9f9,stroke:#6c1fd8
  style C fill:#a9f9d3,stroke:#1fd86c
  style D fill:#a9d3f9,stroke:#1f6cd8
  style E fill:#f9a9d3,stroke:#d81f6c
  style F fill:#d3f9a9,stroke:#6cd81f`;

const DEMO_SEQUENCE_CHART = `sequenceDiagram
  participant User
  participant DeepWiki
  participant GitHub

  User->>DeepWiki: Enter repository URL
  DeepWiki->>GitHub: Request repository data
  GitHub-->>DeepWiki: Return repository data
  DeepWiki->>DeepWiki: Process and analyze code
  DeepWiki-->>User: Display wiki with diagrams

  %% Add a note to make text more visible
  Note over User,GitHub: DeepWiki supports sequence diagrams for visualizing interactions`;

export default function Home() {
  const { messages } = useLanguage();
  const { projects, isLoading: projectsLoading, removeProject } = useProcessedProjects();

  const t = (key: string, params: Record<string, string | number> = {}): string => {
    const keys = key.split('.');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let value: any = messages;
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return key;
      }
    }
    if (typeof value === 'string') {
      return Object.entries(params).reduce((acc: string, [paramKey, paramValue]) => {
        return acc.replace(`{${paramKey}}`, String(paramValue));
      }, value);
    }
    return key;
  };

  return (
    <div className="h-screen paper-texture p-4 md:p-6 flex flex-col relative overflow-hidden">
      {/* Decorative floating orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden z-0">
        <div className="absolute -top-32 -right-32 w-96 h-96 rounded-full bg-[var(--accent-primary)]/8 blur-3xl animate-float" />
        <div className="absolute -bottom-48 -left-48 w-[500px] h-[500px] rounded-full bg-[var(--highlight)]/6 blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/3 right-1/4 w-64 h-64 rounded-full bg-purple-500/5 blur-3xl animate-float" style={{ animationDelay: '4s' }} />
      </div>

      <header className="max-w-6xl mx-auto mb-6 h-fit w-full animate-fade-in relative z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 glass-surface rounded-2xl shadow-elevated p-5">
          <div className="flex items-center">
            <div className="relative">
              <div className="absolute -inset-1 bg-[var(--accent-primary)]/20 rounded-2xl blur-md animate-glow" />
              <div className="relative bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] p-3 rounded-2xl shadow-lg">
                <AzureDevOpsIcon className="text-2xl text-white" />
              </div>
            </div>
            <div className="ml-4 mr-6">
              <h1 className="text-xl md:text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-[var(--accent-primary)] to-[var(--highlight)]">{t('common.appName')}</h1>
              <div className="flex flex-wrap items-baseline gap-x-2 md:gap-x-3 mt-0.5">
                <p className="text-xs text-[var(--muted)] whitespace-nowrap">{t('common.tagline')}</p>
                <div className="hidden md:inline-block">
                  <Link href="/wiki/projects"
                    className="text-xs font-semibold text-[var(--accent-primary)] hover:text-[var(--highlight)] whitespace-nowrap transition-colors px-2 py-0.5 rounded-full hover:bg-[var(--accent-primary)]/10">
                    {t('nav.wikiProjects')}
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 max-w-6xl mx-auto w-full relative z-10">
        <div className="h-full flex flex-col items-center p-8 pt-10 glass-surface rounded-3xl shadow-elevated animate-fade-in" style={{ animationDelay: '0.1s' }}>
          {!projectsLoading && projects.length > 0 ? (
            <div className="w-full flex-1 min-h-0 flex flex-col">
              <div className="flex flex-col items-center w-full max-w-2xl mb-8 mx-auto">
                <div className="flex flex-col sm:flex-row items-center mb-6 gap-5">
                  <div className="relative">
                    <div className="absolute -inset-4 bg-[var(--accent-primary)]/15 rounded-full blur-2xl animate-glow" />
                    <div className="relative bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] p-4 rounded-3xl shadow-elevated">
                      <AzureDevOpsIcon className="text-5xl text-white" />
                    </div>
                  </div>
                  <div className="text-center sm:text-left">
                    <h2 className="text-2xl font-bold text-[var(--foreground)] mb-1">{t('projects.existingProjects')}</h2>
                    <p className="text-[var(--accent-primary)] text-sm max-w-md font-medium">{t('projects.browseExisting')}</p>
                  </div>
                </div>
              </div>

              <ProcessedProjects
                showHeader={false}
                maxItems={6}
                messages={messages}
                className="w-full"
                projects={projects}
                isLoading={projectsLoading}
                onProjectDeleted={removeProject}
              />
            </div>
          ) : (
            <>
              <div className="flex flex-col items-center w-full max-w-2xl mb-8">
                <div className="flex flex-col sm:flex-row items-center mb-6 gap-5">
                  <div className="relative">
                    <div className="absolute -inset-4 bg-[var(--accent-primary)]/15 rounded-full blur-2xl animate-glow" />
                    <div className="relative bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] p-4 rounded-3xl shadow-elevated">
                      <AzureDevOpsIcon className="text-5xl text-white" />
                    </div>
                  </div>
                  <div className="text-center sm:text-left">
                    <h2 className="text-2xl font-bold text-[var(--foreground)] mb-1">{t('home.welcome')}</h2>
                    <p className="text-[var(--accent-primary)] text-sm max-w-md font-medium">{t('home.welcomeTagline')}</p>
                  </div>
                </div>

                <p className="text-[var(--foreground)] text-center mb-8 text-lg leading-relaxed max-w-lg">
                  {t('home.description')}
                </p>
              </div>

              <div className="w-full max-w-2xl mb-10 glass-surface rounded-2xl p-6 border border-[var(--accent-primary)]/15">
                <h3 className="text-sm font-semibold text-[var(--accent-primary)] mb-3 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {t('home.quickStart')}
                </h3>
                <p className="text-sm text-[var(--foreground)] mb-3">
                  Use the code processor to generate wikis:
                </p>
                <div className="grid grid-cols-1 gap-3 text-xs text-[var(--muted)]">
                  <div className="bg-[var(--card-bg-solid)]/70 p-4 rounded-xl border border-[var(--border-color)] font-mono overflow-x-hidden whitespace-nowrap shadow-sm">
                    python -m backend.processor.code_processor --config=backend/run.json
                  </div>
                </div>
              </div>

              <div className="w-full max-w-2xl mb-8 glass-surface rounded-2xl p-6">
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 mb-4">
                  <div className="bg-gradient-to-br from-[var(--accent-primary)] to-[var(--highlight)] p-1.5 rounded-xl">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  <h3 className="text-base font-bold text-[var(--foreground)]">{t('home.advancedVisualization')}</h3>
                </div>
                <p className="text-sm text-[var(--foreground)] mb-5 leading-relaxed">
                  {t('home.diagramDescription')}
                </p>

                <div className="grid grid-cols-1 gap-6">
                  <div className="bg-[var(--card-bg-solid)]/80 p-6 rounded-2xl border border-[var(--border-color)] shadow-custom hover:shadow-elevated transition-all duration-300 hover:border-[var(--accent-primary)]/20 group">
                    <h4 className="text-sm font-semibold text-[var(--foreground)] mb-4 flex items-center gap-2 group-hover:text-[var(--accent-primary)] transition-colors">
                      <span className="w-2 h-2 rounded-full bg-gradient-to-r from-[var(--accent-primary)] to-[var(--highlight)]" />
                      {t('home.flowDiagram')}
                    </h4>
                    <Mermaid chart={DEMO_FLOW_CHART} />
                  </div>
                  <div className="bg-[var(--card-bg-solid)]/80 p-6 rounded-2xl border border-[var(--border-color)] shadow-custom hover:shadow-elevated transition-all duration-300 hover:border-[var(--accent-primary)]/20 group">
                    <h4 className="text-sm font-semibold text-[var(--foreground)] mb-4 flex items-center gap-2 group-hover:text-[var(--accent-primary)] transition-colors">
                      <span className="w-2 h-2 rounded-full bg-gradient-to-r from-[var(--highlight)] to-purple-500" />
                      {t('home.sequenceDiagram')}
                    </h4>
                    <Mermaid chart={DEMO_SEQUENCE_CHART} />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </main>

      <footer className="max-w-6xl mx-auto mt-6 flex flex-col gap-4 w-full relative z-10">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 glass-surface rounded-2xl p-4">
          <div className="flex items-center gap-2 shrink-0">
            <MicrosoftLogo className="h-3.5 w-auto opacity-50" />
            <p className="text-[var(--muted)] text-xs whitespace-nowrap">{t('footer.brand')}</p>
          </div>
          <p className="text-[var(--muted)] text-xs flex-1 text-center opacity-75">{t('footer.copyright')}</p>
          <div className="flex items-center shrink-0">
            <ThemeToggle />
          </div>
        </div>
      </footer>
    </div>
  );
}
