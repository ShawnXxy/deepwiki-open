import React, { useEffect, useRef, useState, useCallback } from 'react';
import mermaid from 'mermaid';
import logger from '../utils/logger';

// Initialize mermaid with minimal config - styles are in globals.css
mermaid.initialize({
  startOnLoad: true,
  theme: 'neutral',
  securityLevel: 'loose',
  suppressErrorRendering: true,
  logLevel: 'error',
  maxTextSize: 100000,
  htmlLabels: true,
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    nodeSpacing: 60,
    rankSpacing: 60,
    padding: 20,
  },
  fontFamily: 'var(--font-geist-sans), "Segoe UI", sans-serif',
  fontSize: 12,
});

// Reusable icon components
const ZoomOutIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" 
       fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    <line x1="8" y1="11" x2="14" y2="11" />
  </svg>
);

const ZoomInIcon = ({ size = 16 }: { size?: number }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24"
       fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    <line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" />
  </svg>
);

const ResetIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
       fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
  </svg>
);

const CloseIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
       fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const WarningIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

interface MermaidProps {
  chart: string;
  className?: string;
  zoomingEnabled?: boolean;
}

// Full screen modal component for diagram viewing
const FullScreenModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}> = ({ isOpen, onClose, children }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);

  // Keyboard and click-outside handlers
  useEffect(() => {
    if (!isOpen) return;
    
    const handleKeyDown = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    const handleOutsideClick = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) onClose();
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleOutsideClick);
    setZoom(1); // Reset zoom on open

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const buttonClass = "text-[var(--foreground)] hover:bg-[var(--accent-primary)]/10 p-2 rounded-md border border-[var(--border-color)] transition-colors";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 p-4">
      <div ref={modalRef} className="bg-[var(--card-bg)] rounded shadow-custom max-w-5xl max-h-[90vh] w-full overflow-hidden flex flex-col card-azure">
        {/* Header with zoom controls */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-color)]">
          <div className="font-medium text-[var(--foreground)]">Diagram View</div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <button onClick={() => setZoom(Math.max(0.5, zoom - 0.1))} className={buttonClass} aria-label="Zoom out">
                <ZoomOutIcon />
              </button>
              <span className="text-sm text-[var(--muted)]">{Math.round(zoom * 100)}%</span>
              <button onClick={() => setZoom(Math.min(2, zoom + 0.1))} className={buttonClass} aria-label="Zoom in">
                <ZoomInIcon />
              </button>
              <button onClick={() => setZoom(1)} className={buttonClass} aria-label="Reset zoom">
                <ResetIcon />
              </button>
            </div>
            <button onClick={onClose} className={buttonClass} aria-label="Close">
              <CloseIcon />
            </button>
          </div>
        </div>
        {/* Content with zoom transform */}
        <div className="overflow-auto p-6 flex-1 flex items-center justify-center bg-[var(--background)]/50">
          <div style={{ transform: `scale(${zoom})`, transformOrigin: 'center center', transition: 'transform 0.3s ease-out' }}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Sanitizes Mermaid diagram content to fix common parsing issues
 * @param content - Raw Mermaid diagram content  
 * @returns Sanitized content safe for Mermaid rendering
 */
const sanitizeMermaidContent = (content: string): string => {
  if (!content) return content;

  let sanitized = content;
  const isFlowchart = /^\s*(graph|flowchart)\s+(TB|TD|BT|RL|LR)/im.test(sanitized);
  const isSequenceDiagram = /^\s*sequenceDiagram/im.test(sanitized);

  // Fix sequence diagram issues
  if (isSequenceDiagram) {
    // Extract all participant declarations
    const participantRegex = /^\s*participant\s+(\w+)(?:\s+as\s+.+)?$/gim;
    const declaredParticipants = new Set<string>();
    let match;
    while ((match = participantRegex.exec(sanitized)) !== null) {
      declaredParticipants.add(match[1]);
    }

    // Find all used participants in messages
    const messageRegex = /^\s*(\w+)\s*(->>?|-->>?|-\)|\)\)|->x|-->x)\s*[+-]?(\w+)/gim;
    const usedParticipants = new Set<string>();
    let msgMatch;
    const tempContent = sanitized;
    while ((msgMatch = messageRegex.exec(tempContent)) !== null) {
      usedParticipants.add(msgMatch[1]); // From participant
      usedParticipants.add(msgMatch[3]); // To participant
    }

    // Add missing participant declarations at the top
    const missingParticipants = Array.from(usedParticipants).filter(p => !declaredParticipants.has(p));
    if (missingParticipants.length > 0) {
      const sequenceDiagramLine = sanitized.match(/^(\s*sequenceDiagram\s*)$/im);
      if (sequenceDiagramLine && sequenceDiagramLine.index !== undefined) {
        const insertPos = sequenceDiagramLine.index + sequenceDiagramLine[0].length;
        const declarations = missingParticipants.map(p => `\n    participant ${p}`).join('');
        sanitized = sanitized.slice(0, insertPos) + declarations + sanitized.slice(insertPos);
      }
    }

    // Fix invalid arrow syntax like )|  - should be -) for async
    sanitized = sanitized.replace(/(\w+)\s*\)\|\s*(\w+)/g, '$1 -) $2');
    
    // Fix PS syntax error - likely meant -) for async or ->> for sync
    sanitized = sanitized.replace(/(\w+)\s*PS\s*(\w+)/g, '$1 -) $2');
  }

  // Convert sequence arrows to flowchart arrows when misused
  if (isFlowchart && !isSequenceDiagram) {
    sanitized = sanitized
      .replace(/(\w+)\s*-->>\s*(\w+)/g, '$1 -.-> $2')
      .replace(/(\w+)\s*->>\s*(\w+)/g, '$1 --> $2')
      .replace(/(\w+)\s*->>([^>])/g, '$1 -->$2')
      .replace(/(\w+)\s*-->>([^>])/g, '$1 -.->$2');
  }

  // Fix edge labels - remove special chars
  sanitized = sanitized.replace(/-->\|([^|]+)\|/g, (_, label) => 
    `-->|${label.replace(/[<>]/g, '').trim()}|`
  );

  // Fix parentheses in square bracket labels
  sanitized = sanitized.replace(/(\w+)\[([^\]]*\([^)]*\)[^\]]*)\]/g, (_, nodeId, label) => 
    `${nodeId}["${label.replace(/\(/g, '❨').replace(/\)/g, '❩')}"]`
  );

  // Fix commas in node labels
  sanitized = sanitized.replace(/(\w+)\[([^\]]*,[^\]]*)\]/g, (match, nodeId, label) => {
    if (label.startsWith('"') && label.endsWith('"')) return match;
    return `${nodeId}["${label.replace(/,/g, ';')}"]`;
  });

  // Convert source citations to comments
  sanitized = sanitized.replace(/Sources:\s*\[([^\]]+)\]\(\)/g, '%% Source: $1');
  sanitized = sanitized.replace(/Sources:\s*(\[([^\]]+)\]\([^)]+\)(?:,\s*)?)+/g, (match) => {
    const citations: string[] = [];
    let m;
    const pattern = /\[([^\]]+)\]\(([^)]+)\)/g;
    while ((m = pattern.exec(match)) !== null) {
      citations.push(`%% Source: ${m[1]} - ${m[2]}`);
    }
    return citations.join('\n');
  });

  // Fix standalone brackets with empty URLs
  sanitized = sanitized.replace(/\[([^\]]*)\]\(\)(?!\s*-->|\s*---|\s*--)/g, '($1)');

  // Fix nested parentheses in round-bracket nodes
  sanitized = sanitized.replace(/(\w+)\(([^)]*\([^)]*\)[^)]*)\)/g, (_, nodeId, label) => 
    `${nodeId}(${label.replace(/\(/g, '❨').replace(/\)/g, '❩')})`
  );

  if (content !== sanitized) {
    logger.debug('Mermaid content sanitized', {
      original: content.substring(0, 200),
      sanitized: sanitized.substring(0, 200)
    });
  }

  return sanitized;
};

const Mermaid: React.FC<MermaidProps> = ({ chart, className = '', zoomingEnabled = false }) => {
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const mermaidRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(`mermaid-${Math.random().toString(36).substring(2, 9)}`);
  const isDarkMode = typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches;

  // Initialize pan-zoom for zoomable diagrams
  useEffect(() => {
    if (!svg || !zoomingEnabled || !containerRef.current) return;

    const initPanZoom = async () => {
      const svgEl = containerRef.current?.querySelector('svg');
      if (!svgEl) return;

      svgEl.style.maxWidth = 'none';
      svgEl.style.width = '100%';
      svgEl.style.height = '100%';

      try {
        const svgPanZoom = (await import('svg-pan-zoom')).default;
        svgPanZoom(svgEl, {
          zoomEnabled: true,
          controlIconsEnabled: true,
          fit: true,
          center: true,
          minZoom: 0.1,
          maxZoom: 10,
          zoomScaleSensitivity: 0.3,
        });
      } catch (err) {
        logger.error('Failed to load svg-pan-zoom', { error: String(err) });
      }
    };

    setTimeout(initPanZoom, 100);
  }, [svg, zoomingEnabled]);

  // Render chart when content changes
  useEffect(() => {
    if (!chart) return;
    let isMounted = true;

    const renderChart = async () => {
      if (!isMounted) return;

      try {
        setError(null);
        setSvg('');

        const sanitizedChart = sanitizeMermaidContent(chart);
        const { svg: rendered } = await mermaid.render(idRef.current, sanitizedChart);

        if (!isMounted) return;

        // Add dark mode attribute
        let processed = rendered;
        if (isDarkMode) {
          processed = processed.replace('<svg ', '<svg data-theme="dark" ');
        }

        setSvg(processed);
        setTimeout(() => mermaid.contentLoaded(), 50);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        // Use warn instead of error — Mermaid syntax issues are expected
        // and handled gracefully in the UI with a fallback display.
        logger.warn('Mermaid rendering issue', {
          chart: chart.substring(0, 200)
        });

        if (isMounted) {
          setError(`Failed to render diagram: ${errorMsg}`);
          if (mermaidRef.current) {
            const sanitized = sanitizeMermaidContent(chart);
            mermaidRef.current.innerHTML = `
              <div class="text-red-500 dark:text-red-400 text-xs mb-1">Syntax error in diagram</div>
              <details class="text-xs mb-2">
                <summary class="cursor-pointer text-gray-600 dark:text-gray-400">Show original</summary>
                <pre class="text-xs overflow-auto p-2 bg-gray-100 dark:bg-gray-800 rounded mt-1">${chart}</pre>
              </details>
              <details class="text-xs">
                <summary class="cursor-pointer text-gray-600 dark:text-gray-400">Show sanitized</summary>
                <pre class="text-xs overflow-auto p-2 bg-gray-100 dark:bg-gray-800 rounded mt-1">${sanitized}</pre>
              </details>
            `;
          }
        }
      }
    };

    renderChart();
    return () => { isMounted = false; };
  }, [chart, isDarkMode]);

  const handleDiagramClick = useCallback(() => {
    if (!error && svg) setIsFullscreen(true);
  }, [error, svg]);

  // Error state
  if (error) {
    return (
      <div className={`border border-[var(--highlight)]/30 rounded-md p-4 bg-[var(--highlight)]/5 ${className}`}>
        <div className="flex items-center mb-3">
          <div className="text-[var(--highlight)] text-xs font-medium flex items-center">
            <WarningIcon />
            Diagram Rendering Error
          </div>
        </div>
        <div ref={mermaidRef} className="text-xs overflow-auto" />
        <div className="mt-3 text-xs text-[var(--muted)]">
          The diagram contains syntax errors and cannot be rendered.
        </div>
      </div>
    );
  }

  // Loading state
  if (!svg) {
    return (
      <div className={`flex justify-center items-center p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse" />
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-75" />
          <div className="w-2 h-2 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-150" />
          <span className="text-[var(--muted)] text-xs ml-2">Rendering diagram...</span>
        </div>
      </div>
    );
  }

  // Rendered diagram
  return (
    <>
      <div ref={containerRef} className={`w-full max-w-full ${zoomingEnabled ? 'h-[600px] p-4' : ''}`}>
        <div className={`relative group ${zoomingEnabled ? 'h-full rounded-lg border-2 border-black' : ''}`}>
          <div
            className={`mermaid-diagram flex justify-center overflow-auto text-center my-2 cursor-pointer hover:shadow-md transition-shadow duration-200 rounded-md ${className} ${zoomingEnabled ? 'h-full' : ''}`}
            dangerouslySetInnerHTML={{ __html: svg }}
            onClick={zoomingEnabled ? undefined : handleDiagramClick}
            title={zoomingEnabled ? undefined : 'Click to view fullscreen'}
          />
          {!zoomingEnabled && (
            <div className="absolute top-2 right-2 bg-gray-700/70 dark:bg-gray-900/70 text-white p-1.5 rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1.5 text-xs shadow-md pointer-events-none">
              <ZoomInIcon size={12} />
              <span>Click to zoom</span>
            </div>
          )}
        </div>
      </div>
      {!zoomingEnabled && (
        <FullScreenModal isOpen={isFullscreen} onClose={() => setIsFullscreen(false)}>
          <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: svg }} />
        </FullScreenModal>
      )}
    </>
  );
};



export default Mermaid;