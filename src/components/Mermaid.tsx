import React, { useEffect, useRef, useState, useCallback } from 'react';
import mermaid from 'mermaid';
import logger from '../utils/logger';

// Initialize mermaid with minimal config - styles are in globals.css
mermaid.initialize({
  startOnLoad: false,
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
    padding: 30,
    useMaxWidth: false,
    wrappingWidth: 200,
  },
  fontFamily: '"Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif',
  fontSize: 14,
} as Parameters<typeof mermaid.initialize>[0]);

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
 * Sanitize mermaid content with progressive levels of aggressiveness.
 *
 * Level 0 — Structural fixes only (mismatched diagram types, missing
 *           participants, source citations). Safe for all diagrams.
 * Level 1 — Escape special characters in labels and edge text.
 *           Fixes the majority of LLM-generated syntax errors.
 * Level 2 — Strip all advanced features that LLMs frequently break
 *           (activation markers, nested shapes). Last resort before
 *           falling back to a code-block display.
 */
function sanitizeAtLevel(content: string, level: number): string {
  if (!content) return content;

  let s = content;
  const isFlowchart = /^\s*(graph|flowchart)\s+(TB|TD|BT|RL|LR)/im.test(s);
  const isSequence = /^\s*sequenceDiagram/im.test(s);

  // ── Level 0: structural / cross-syntax fixes (always applied) ──────────

  // Convert source citations to mermaid comments
  s = s.replace(/Sources:\s*\[([^\]]+)\]\(\)/g, '%% Source: $1');
  s = s.replace(/Sources:\s*(\[([^\]]+)\]\([^)]+\)(?:,\s*)?)+/g, (match) => {
    const citations: string[] = [];
    let m;
    const p = /\[([^\]]+)\]\(([^)]+)\)/g;
    while ((m = p.exec(match)) !== null) citations.push(`%% Source: ${m[1]} - ${m[2]}`);
    return citations.join('\n');
  });
  s = s.replace(/\[([^\]]*)\]\(\)(?!\s*-->|\s*---|\s*--)/g, '($1)');

  if (isSequence) {
    // Auto-declare missing participants
    const declared = new Set<string>();
    for (const m of s.matchAll(/^\s*participant\s+(\w+)/gim)) declared.add(m[1]);
    const used = new Set<string>();
    for (const m of s.matchAll(/^\s*(\w+)\s*(->>?|-->>?|-\)|\)\)|->x|-->x)\s*[+-]?(\w+)/gim)) {
      used.add(m[1]); used.add(m[3]);
    }
    const missing = [...used].filter(p => !declared.has(p));
    if (missing.length) {
      const anchor = s.match(/^(\s*sequenceDiagram\s*)$/im);
      if (anchor?.index !== undefined) {
        const pos = anchor.index + anchor[0].length;
        s = s.slice(0, pos) + missing.map(p => `\n    participant ${p}`).join('') + s.slice(pos);
      }
    }
    // Fix obviously wrong arrow tokens
    s = s.replace(/(\w+)\s*\)\|\s*(\w+)/g, '$1 -) $2');
    s = s.replace(/(\w+)\s*PS\s*(\w+)/g, '$1 -) $2');
  }

  if (isFlowchart && !isSequence) {
    // Convert misused sequence arrows in flowcharts
    s = s.replace(/(\w+)\s*-->>\s*(\w+)/g, '$1 -.-> $2');
    s = s.replace(/(\w+)\s*->>\s*(\w+)/g, '$1 --> $2');
    s = s.replace(/(\w+)\s*->>([^>])/g, '$1 -->$2');
    s = s.replace(/(\w+)\s*-->>([^>])/g, '$1 -.->$2');
    // Convert sequence-style colon labels to pipe labels
    s = s.replace(/(\w+)\s*(-->|-.->)\s*(\w+):\s*(.+)$/gm, '$1 $2|$4| $3');
  }

  if (level === 0) return s;

  // ── Level 1: escape special chars in labels and messages ───────────────

  if (isSequence) {
    // Curly braces in message text
    s = s.replace(/(:\s*[^:\n]*)\{([^}\n]*)\}/g, '$1($2)');
  }

  if (isFlowchart || (!isSequence && !isFlowchart)) {
    // Escape edge labels (pipe-delimited text between arrows)
    const escapeLabel = (_: string, label: string) =>
      label.replace(/[<>]/g, '').replace(/\(/g, '❨').replace(/\)/g, '❩').trim();
    s = s.replace(/-->\|([^|]+)\|/g, (m, l) => `-->|${escapeLabel(m, l)}|`);
    s = s.replace(/-\.->\|([^|]+)\|/g, (m, l) => `-.->|${escapeLabel(m, l)}|`);

    // Parentheses inside square-bracket labels
    s = s.replace(/(\w+)\[([^\]]*\([^)]*\)[^\]]*)\]/g, (_, id, l) =>
      `${id}["${l.replace(/\(/g, '❨').replace(/\)/g, '❩')}"]`
    );

    // Nested square brackets inside labels
    s = s.replace(/(\w+)\[("?)([^\]"]*)\[([^\]]*)\]([^\]"]*)\2\]/g,
      (_, id, _q, a, inner, b) => `${id}["${a}⟦${inner}⟧${b}"]`
    );

    // Commas in unquoted labels
    s = s.replace(/(\w+)\[([^\]]*,[^\]]*)\]/g, (match, id, l) => {
      if (l.startsWith('"') && l.endsWith('"')) return match;
      return `${id}["${l.replace(/,/g, ';')}"]`;
    });

    // Nested parentheses in round-bracket nodes
    s = s.replace(/(\w+)\(([^)]*\([^)]*\)[^)]*)\)/g, (_, id, l) =>
      `${id}(${l.replace(/\(/g, '❨').replace(/\)/g, '❩')})`
    );
  }

  if (level === 1) return s;

  // ── Level 2: strip advanced features that LLMs frequently break ────────

  if (isSequence) {
    // Strip activation markers (+/-) — LLMs mismatch them constantly
    s = s.replace(/(->>?|-->>?)\s*([+-])\s*/g, '$1 ');
    s = s.replace(/(\w+)\s*([+-])\s*(->>?|-->>?)/g, '$1 $3');
  }

  // Quote ALL unquoted square-bracket labels as a nuclear option
  s = s.replace(/(\w+)\[([^\]"]+)\]/g, (match, id, label) => {
    if (/^[a-zA-Z0-9 _.:-]+$/.test(label)) return match; // already safe
    return `${id}["${label.replace(/"/g, '#quot;')}"]`;
  });

  return s;
}

const Mermaid: React.FC<MermaidProps> = ({ chart, className = '', zoomingEnabled = false }) => {
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
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
      svgEl.style.overflow = 'visible';

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

  // Render chart with progressive sanitization: try increasing levels
  // until one succeeds, or fall back to displaying source code.
  useEffect(() => {
    if (!chart) return;
    let isMounted = true;
    // Each attempt needs a unique ID to avoid mermaid ID collisions
    const baseId = idRef.current;

    const tryRender = async (text: string, attempt: number): Promise<string | null> => {
      // Validate syntax first — parse() with suppressErrors silently returns
      // false without triggering console.error or Next.js error overlay.
      // Only proceed to render() when we know the syntax is valid.
      const parseResult = await mermaid.parse(text, { suppressErrors: true });
      if (!parseResult) return null;

      try {
        const id = attempt === 0 ? baseId : `${baseId}-r${attempt}`;
        const { svg } = await mermaid.render(id, text);
        return svg;
      } catch {
        return null;
      }
    };

    const renderChart = async () => {
      if (!isMounted) return;
      setError(null);
      setSvg('');

      // Progressive rendering: try level 0 → 1 → 2
      for (let level = 0; level <= 2; level++) {
        if (!isMounted) return;
        const sanitized = sanitizeAtLevel(chart, level);
        const rendered = await tryRender(sanitized, level);

        if (rendered && isMounted) {
          let processed = rendered;
          if (isDarkMode) {
            processed = processed.replace('<svg ', '<svg data-theme="dark" ');
          }
          setSvg(processed);
          if (level > 0) {
            logger.debug('Mermaid rendered after sanitization', { level, chart: chart.substring(0, 100) });
          }
          return;
        }
      }

      // All levels failed — show source as fallback
      if (isMounted) {
        logger.warn('Mermaid rendering failed at all levels', { chart: chart.substring(0, 200) });
        setError('Diagram could not be rendered');
      }
    };

    renderChart();
    return () => { isMounted = false; };
  }, [chart, isDarkMode]);

  const handleDiagramClick = useCallback(() => {
    if (!error && svg) setIsFullscreen(true);
  }, [error, svg]);

  // Error fallback — show source as formatted code block instead of ugly error
  if (error) {
    return (
      <div className={`my-4 rounded-md overflow-hidden text-sm shadow-sm ${className}`}>
        <div className="bg-gray-800 text-gray-200 px-5 py-2 text-sm flex justify-between items-center">
          <span className="text-[var(--muted)] text-xs">mermaid (source)</span>
        </div>
        <pre className="text-xs overflow-auto p-4 bg-gray-900 text-gray-300 leading-relaxed">{chart}</pre>
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