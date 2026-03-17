import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Mermaid from './Mermaid';

interface MarkdownProps {
  content: string;
  onNavigateToPage?: (pageId: string) => void;
}

/**
 * Wrap unfenced Mermaid diagram blocks in ```mermaid code fences.
 *
 * The LLM sometimes outputs Mermaid diagrams without code fences, e.g.:
 *   graph TD
 *     A --> B
 *
 * This pre-processor detects lines that start a Mermaid diagram and
 * wraps the contiguous block in ```mermaid ... ``` so the code
 * component can render them via the Mermaid component.
 */
function wrapUnfencedMermaid(md: string): string {
  // Patterns that indicate the start of a Mermaid diagram
  const mermaidStartPatterns = [
    /^(graph|flowchart)\s+(TD|TB|BT|RL|LR)\b/,
    /^sequenceDiagram\b/,
    /^classDiagram\b/,
    /^erDiagram\b/,
    /^stateDiagram(?:-v2)?\b/,
    /^gantt\b/,
    /^pie\b/,
    /^gitgraph\b/,
    /^journey\b/,
    /^C4Context\b/,
  ];

  const lines = md.split('\n');
  const result: string[] = [];
  let inFence = false;       // inside any ``` code fence
  let inMermaid = false;     // inside an unfenced mermaid block we're wrapping
  let blankCount = 0;        // consecutive blank lines while in mermaid block

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trimStart();

    // Track code fences — don't touch anything inside existing fences
    if (trimmed.startsWith('```')) {
      if (!inFence) {
        // Entering a fence — close any open mermaid block first
        if (inMermaid) {
          result.push('```');
          inMermaid = false;
        }
        inFence = true;
      } else {
        inFence = false;
      }
      result.push(line);
      continue;
    }

    if (inFence) {
      result.push(line);
      continue;
    }

    // Check if this line starts a Mermaid diagram (not inside a fence)
    if (!inMermaid && mermaidStartPatterns.some(p => p.test(trimmed))) {
      result.push('```mermaid');
      result.push(line);
      inMermaid = true;
      blankCount = 0;
      continue;
    }

    if (inMermaid) {
      // Mermaid blocks end when we hit 2+ consecutive blank lines,
      // or a line that looks like regular markdown (heading, list, paragraph text)
      if (trimmed === '') {
        blankCount++;
        if (blankCount >= 2) {
          result.push('```');
          result.push('');  // keep one blank line
          inMermaid = false;
          continue;
        }
        result.push(line);
        continue;
      }
      blankCount = 0;

      // Lines that signal end of diagram: markdown prose, not mermaid syntax
      if (/^#{1,6}\s/.test(trimmed) ||          // headings
          /^---+$/.test(trimmed) ||              // horizontal rules
          trimmed.startsWith('<') ||              // HTML tags
          trimmed.startsWith('Sources:') ||       // citation lines
          trimmed.startsWith('*') ||              // italic/bold/list
          trimmed.startsWith('-') && !trimmed.includes('->') && !trimmed.includes('-->') || // list items (not arrows)
          /^\d+\.\s/.test(trimmed) ||            // numbered lists
          trimmed.startsWith('|')) {              // table rows
        result.push('```');
        result.push(line);
        inMermaid = false;
        continue;
      }

      // Otherwise, still inside the diagram
      result.push(line);
      continue;
    }

    result.push(line);
  }

  // Close any open mermaid block at end of content
  if (inMermaid) {
    result.push('```');
  }

  return result.join('\n');
}

const Markdown: React.FC<MarkdownProps> = ({ content, onNavigateToPage }) => {
  // Pre-process: wrap unfenced Mermaid diagrams in code fences
  const processedContent = React.useMemo(() => wrapUnfencedMermaid(content), [content]);
  // Define markdown components
  const MarkdownComponents: React.ComponentProps<typeof ReactMarkdown>['components'] = {
    p({ children, ...props }: { children?: React.ReactNode }) {
      return <p className="mb-3 text-sm leading-relaxed dark:text-white" {...props}>{children}</p>;
    },
    h1({ children, ...props }: { children?: React.ReactNode }) {
      const headingId = children && typeof children === 'string'
        ? children.toString().toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
        : undefined;
      return <h1 id={headingId} className="text-xl font-bold mt-6 mb-3 dark:text-white" {...props}>{children}</h1>;
    },
    h2({ children, ...props }: { children?: React.ReactNode }) {
      // Generate anchor ID from heading text for in-page links
      const headingId = children && typeof children === 'string'
        ? children.toString().toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
        : undefined;
      // Special styling for ReAct headings
      if (children && typeof children === 'string') {
        const text = children.toString();
        if (text.includes('Thought') || text.includes('Action') || text.includes('Observation') || text.includes('Answer')) {
          return (
            <h2
              id={headingId}
              className={`text-base font-bold mt-5 mb-3 p-2 rounded ${
                text.includes('Thought') ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300' :
                text.includes('Action') ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                text.includes('Observation') ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300' :
                text.includes('Answer') ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300' :
                'dark:text-white'
              }`}
              {...props}
            >
              {children}
            </h2>
          );
        }
      }
      return <h2 id={headingId} className="text-lg font-bold mt-5 mb-3 dark:text-white" {...props}>{children}</h2>;
    },
    h3({ children, ...props }: { children?: React.ReactNode }) {
      const headingId = children && typeof children === 'string'
        ? children.toString().toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
        : undefined;
      return <h3 id={headingId} className="text-base font-semibold mt-4 mb-2 dark:text-white" {...props}>{children}</h3>;
    },
    h4({ children, ...props }: { children?: React.ReactNode }) {
      return <h4 className="text-sm font-semibold mt-3 mb-2 dark:text-white" {...props}>{children}</h4>;
    },
    ul({ children, ...props }: { children?: React.ReactNode }) {
      return <ul className="list-disc pl-6 mb-4 text-sm dark:text-white space-y-2" {...props}>{children}</ul>;
    },
    ol({ children, ...props }: { children?: React.ReactNode }) {
      return <ol className="list-decimal pl-6 mb-4 text-sm dark:text-white space-y-2" {...props}>{children}</ol>;
    },
    li({ children, ...props }: { children?: React.ReactNode }) {
      return <li className="mb-2 text-sm leading-relaxed dark:text-white" {...props}>{children}</li>;
    },
    a({ children, href, ...props }: { children?: React.ReactNode; href?: string }) {
      // Cross-page wiki links: deepwiki://page_id
      if (href?.startsWith('deepwiki://') && onNavigateToPage) {
        const pageId = href.replace('deepwiki://', '');
        return (
          <a
            href="#"
            className="text-purple-600 dark:text-purple-400 hover:underline font-medium"
            onClick={(e) => {
              e.preventDefault();
              onNavigateToPage(pageId);
            }}
            {...props}
          >
            {children}
          </a>
        );
      }
      // In-page anchor links: scroll to heading, don't open new tab
      if (href?.startsWith('#')) {
        return (
          <a
            href={href}
            className="text-purple-600 dark:text-purple-400 hover:underline font-medium"
            onClick={(e) => {
              e.preventDefault();
              const targetId = href.slice(1);
              const el = document.getElementById(targetId);
              if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
            }}
            {...props}
          >
            {children}
          </a>
        );
      }
      return (
        <a
          href={href}
          className="text-purple-600 dark:text-purple-400 hover:underline font-medium"
          target="_blank"
          rel="noopener noreferrer"
          {...props}
        >
          {children}
        </a>
      );
    },
    blockquote({ children, ...props }: { children?: React.ReactNode }) {
      return (
        <blockquote
          className="border-l-4 border-gray-300 dark:border-gray-700 pl-4 py-1 text-gray-700 dark:text-gray-300 italic my-4 text-sm"
          {...props}
        >
          {children}
        </blockquote>
      );
    },
    table({ children, ...props }: { children?: React.ReactNode }) {
      return (
        <div className="overflow-x-auto my-6 rounded-md">
          <table className="min-w-full text-sm border-collapse" {...props}>
            {children}
          </table>
        </div>
      );
    },
    thead({ children, ...props }: { children?: React.ReactNode }) {
      return <thead className="bg-gray-100 dark:bg-gray-800" {...props}>{children}</thead>;
    },
    tbody({ children, ...props }: { children?: React.ReactNode }) {
      return <tbody className="divide-y divide-gray-200 dark:divide-gray-700" {...props}>{children}</tbody>;
    },
    tr({ children, ...props }: { children?: React.ReactNode }) {
      return <tr className="hover:bg-gray-50 dark:hover:bg-gray-900" {...props}>{children}</tr>;
    },
    th({ children, ...props }: { children?: React.ReactNode }) {
      return (
        <th
          className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300"
          {...props}
        >
          {children}
        </th>
      );
    },
    td({ children, ...props }: { children?: React.ReactNode }) {
      return <td className="px-4 py-3 border-t border-gray-200 dark:border-gray-700" {...props}>{children}</td>;
    },
    code(props: {
      inline?: boolean;
      className?: string;
      children?: React.ReactNode;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      [key: string]: any; // Using any here as it's required for ReactMarkdown components
    }) {
      const { inline, className, children, ...otherProps } = props;
      const match = /language-(\w+)/.exec(className || '');
      const codeContent = children ? String(children).replace(/\n$/, '') : '';

      // Handle Mermaid diagrams
      if (!inline && match && match[1] === 'mermaid') {
        return (
          <div className="my-8 bg-gray-50 dark:bg-gray-800 rounded-md overflow-hidden shadow-sm">
            <Mermaid
              chart={codeContent}
              className="w-full max-w-full"
              zoomingEnabled={true}
            />
          </div>
        );
      }

      // Handle code blocks
      if (!inline && match) {
        return (
          <div className="my-6 rounded-md overflow-hidden text-sm shadow-sm">
            <div className="bg-gray-800 text-gray-200 px-5 py-2 text-sm flex justify-between items-center">
              <span>{match[1]}</span>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(codeContent);
                }}
                className="text-gray-400 hover:text-white"
                title="Copy code"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              </button>
            </div>
            <SyntaxHighlighter
              language={match[1]}
              style={tomorrow}
              className="!text-sm"
              customStyle={{ margin: 0, borderRadius: '0 0 0.375rem 0.375rem', padding: '1rem' }}
              showLineNumbers={true}
              wrapLines={true}
              wrapLongLines={true}
              {...otherProps}
            >
              {codeContent}
            </SyntaxHighlighter>
          </div>
        );
      }

      // Handle inline code
      return (
        <code
          className={`${className} font-mono bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded text-pink-500 dark:text-pink-400 text-sm`}
          {...otherProps}
        >
          {children}
        </code>
      );
    },
  };

  return (
    <div className="prose prose-base dark:prose-invert max-w-none px-2 py-4">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={MarkdownComponents}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;