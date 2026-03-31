# Frontend Design Document

> Architecture reference for the DeepWiki (Orcas CodeWiki) frontend.
> Last updated: March 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Page Routes](#2-page-routes)
3. [Component Reference](#3-component-reference)
4. [Wiki Viewer Layout](#4-wiki-viewer-layout)
5. [Data Flow: Wiki Loading](#5-data-flow-wiki-loading)
6. [Markdown Rendering Pipeline](#6-markdown-rendering-pipeline)
7. [Mermaid Diagram Rendering](#7-mermaid-diagram-rendering)
8. [Navigation & Scroll Behavior](#8-navigation--scroll-behavior)
9. [Chat / Ask System](#9-chat--ask-system)
10. [API Routes](#10-api-routes)
11. [Contexts & Hooks](#11-contexts--hooks)
12. [Utilities](#12-utilities)
13. [CSS Architecture](#13-css-architecture)
14. [Type Definitions](#14-type-definitions)
15. [Known Constraints](#15-known-constraints)

---

## 1. Architecture Overview

The frontend is a **Next.js 15 App Router** application that reads pre-generated wiki JSON from disk or Azure Blob Storage. There is no server-side wiki generation — the frontend is a **read-only viewer** backed by cached JSON files.

```
┌─────────────────────────────────────────────────────┐
│  Browser                                            │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Sidebar   │  │ Content  │  │ Chat Panel       │  │
│  │ WikiTree  │  │ Markdown │  │ Ask (WebSocket)  │  │
│  │ View      │  │ Mermaid  │  │                  │  │
│  └───────────┘  └──────────┘  └──────────────────┘  │
│         │              │               │            │
│         └──────────────┼───────────────┘            │
│                        ▼                            │
│              /api/wiki_cache (GET)                   │
│                        │                            │
│     ┌──────────────────┼──────────────────┐         │
│     ▼                  ▼                  ▼         │
│  Local disk      Backend HTTP       Azure Blob      │
│  (~/.adalflow)   (:8001)            (Cloud)         │
└─────────────────────────────────────────────────────┘
```

**Key Principle:** The frontend is **self-contained**. It can serve wikis without a running backend — only the Chat/Ask feature requires the FastAPI backend.

---

## 2. Page Routes

| Route | File | Purpose |
|-------|------|---------|
| `/` | `app/page.tsx` | Home — project list, demo charts, wiki generation form |
| `/[owner]/[repo]` | `app/[owner]/[repo]/page.tsx` | Wiki viewer (main page) |
| `/[owner]/[repo]/slides` | `app/[owner]/[repo]/slides/page.tsx` | Slide presentation mode |
| `/[owner]/[repo]/workshop` | `app/[owner]/[repo]/workshop/page.tsx` | Workshop/tutorial mode |
| `/wiki/projects` | `app/wiki/projects/page.tsx` | Browse all cached projects |

---

## 3. Component Reference

### Core Content Components

| Component | File | Props | Description |
|-----------|------|-------|-------------|
| `Markdown` | `Markdown.tsx` | `content`, `onNavigateToPage` | Renders wiki markdown with custom heading IDs, link routing, syntax highlighting, and embedded Mermaid diagrams |
| `Mermaid` | `Mermaid.tsx` | `chart`, `className`, `zoomingEnabled` | Renders mermaid diagram with 3-level progressive sanitization, SVG pan-zoom, and fullscreen modal |
| `WikiTreeView` | `WikiTreeView.tsx` | `wikiStructure`, `currentPageId`, `onPageSelect`, `messages` | Sidebar navigation tree with expandable sections, section IDs, and search |
| `Ask` | `Ask.tsx` | `repoInfo`, `provider`, `model`, `language`, `onRef` | Chat Q&A with WebSocket streaming, deep research mode, and conversation history |

### UI Components

| Component | File | Description |
|-----------|------|-------------|
| `ProcessedProjects` | `ProcessedProjects.tsx` | Grid/list of previously generated wikis with delete |
| `ConfigurationModal` | `ConfigurationModal.tsx` | Wiki generation settings (repo, language, model) |
| `ModelSelectionModal` | `ModelSelectionModal.tsx` | LLM provider/model picker |
| `TokenInput` | `TokenInput.tsx` | Secure Git token input with show/hide |
| `UserSelector` | `UserSelector.tsx` | Organization/user dropdown |
| `WikiTypeSelector` | `WikiTypeSelector.tsx` | Comprehensive vs concise toggle |
| `ThemeToggle` | `theme-toggle.tsx` | Light/dark mode switch |
| `AzureIcon` | `AzureIcon.tsx` | Azure DevOps and Microsoft logo SVGs |

### Generation Progress Components

| Component | File | Description |
|-----------|------|-------------|
| `BackgroundGenerationManager` | `BackgroundGenerationManager.tsx` | Tracks long-running wiki generation |
| `FloatingProgressWidget` | `FloatingProgressWidget.tsx` | Minimizable progress overlay |
| `CompletionNotificationModal` | `CompletionNotificationModal.tsx` | Generation success notification |

---

## 4. Wiki Viewer Layout

The wiki viewer (`[owner]/[repo]/page.tsx`) uses a 3-column layout on desktop:

```
┌──────────────────────────────────────────────────────────┐
│ Header (breadcrumb, home link)                           │
├────────────┬──────────────────────────┬──────────────────┤
│            │                          │                  │
│  Sidebar   │     #wiki-content        │   Chat Panel     │
│  280-320px │     (flex-grow)          │   (1/3 or 12px)  │
│            │                          │                  │
│  • Title   │  Page Title  [Share]     │   Ask component  │
│  • Desc    │                          │                  │
│  • Repo    │  Markdown content        │                  │
│  • Branch  │    └─ Mermaid diagrams   │                  │
│  • Export  │                          │                  │
│  • Search  │  Related Pages           │                  │
│  • Tree    │                          │                  │
│            │                          │                  │
├────────────┴──────────────────────────┴──────────────────┤
│ Footer                                                   │
└──────────────────────────────────────────────────────────┘
```

### CSS Height Chain (Critical)

Every level must propagate height for `overflow-y: auto` to work:

```
body { overflow: hidden }                    ← prevents phantom scrollbar
  └─ div.h-screen.flex.flex-col              ← 100vh viewport
    ├─ header (h-fit)
    ├─ main.flex-1.overflow-hidden           ← fills remaining height
    │  └─ div.h-full.flex.lg:flex-row        ← wiki viewer
    │    ├─ div.h-full.overflow-hidden       ← wiki section (card-azure)
    │    │  ├─ div.h-full.overflow-y-auto    ← sidebar (scrolls independently)
    │    │  └─ div#wiki-content.flex-grow.overflow-y-auto ← content
    │    └─ div.h-full                       ← chat panel
    └─ footer
```

### State Management

```typescript
// Core state
currentPageId         // Which page is displayed
wikiStructure         // Sections/pages metadata (WikiStructure)
generatedPages        // Record<pageId, WikiPage> with content

// UI state
isChatPanelCollapsed  // Chat panel visibility
isSearchOpen          // Search box
searchQuery           // Search filter text

// Cache metadata
commitHash            // Git commit of indexed code
indexedAt             // Timestamp
cachedProvider        // LLM provider used
cachedModel           // LLM model used
effectiveRepoInfo     // Merged repo info from URL + cache
```

---

## 5. Data Flow: Wiki Loading

```
1. URL parsed: owner, repo, language, branch, comprehensive
                    │
2. GET /api/wiki_cache?owner=...&repo=...&language=...
                    │
3. API route tries:
   a. HTTP GET to backend (:8001) — if running
   b. Local file read from ~/.adalflow/wikicache/
   c. Return 404 if not found
                    │
4. Response: {
     wiki_structure,    // WikiStructure with sections/pages
     generated_pages,   // Record<id, WikiPage>
     commit_hash,       // git SHA
     indexed_at,        // ISO timestamp
     provider,          // "azure_openai"
     model,             // "gpt-4o"
     repo               // { owner, repo, type, branch, repoUrl }
   }
                    │
5. State updated:
   setWikiStructure(data.wiki_structure)
   setGeneratedPages(data.generated_pages)
   setCurrentPageId(initialPageId || firstPage)
                    │
6. Components render:
   WikiTreeView ← wikiStructure
   Markdown     ← generatedPages[currentPageId].content
   Ask          ← effectiveRepoInfo
```

### Wiki Cache File Naming Convention

```
deepwiki_cache_{type}_{owner}_{repo}_{lang}_{mode}_{version}-{branch}.json
```

Example: `deepwiki_cache_azuredevops_msdata_orcasql-mysql_en_comprehensive_5.7-master.json`

---

## 6. Markdown Rendering Pipeline

### Component: `Markdown.tsx`

Uses `react-markdown` with `remark-gfm` (tables, strikethrough) and `rehype-raw` (inline HTML).

### Pre-processing: Unfenced Mermaid Detection

LLMs sometimes output mermaid diagrams without code fences. The `wrapUnfencedMermaid()` function detects and wraps them:

```
Input:                          Output:
graph TD                        ```mermaid
  A --> B                       graph TD
  B --> C                         A --> B
                                  B --> C
                                ```
```

**Detection patterns:** `graph`, `flowchart`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `stateDiagram`, `gantt`, `pie`, `gitgraph`, `journey`, `C4Context`

**Termination:** 2+ consecutive blank lines, or markdown prose (headings, lists, HR, HTML tags, source citations, table rows).

### Custom Element Rendering

| Element | Behavior |
|---------|----------|
| **Headings** (h1–h4) | Auto-generate `id` from text: lowercase → strip special chars → replace spaces with hyphens. Example: `"Getting Started!"` → `id="getting-started"` |
| **Links** `deepwiki://page_id` | Cross-page navigation: calls `onNavigateToPage(pageId)` |
| **Links** `#anchor` | In-page scroll: `document.getElementById(id).scrollIntoView()` |
| **Links** `https://...` | External: `target="_blank" rel="noopener noreferrer"` |
| **Code** `` ```mermaid `` | Rendered via `<Mermaid chart={...} zoomingEnabled={true} />` |
| **Code** `` ```lang `` | Syntax highlighted via `react-syntax-highlighter` (tomorrow theme) with line numbers and copy button |
| **Tables** | Wrapped in `overflow-x-auto` for horizontal scroll |

---

## 7. Mermaid Diagram Rendering

### Component: `Mermaid.tsx`

### Architecture: Progressive Sanitization with Retry

Instead of applying all sanitization rules at once, the renderer tries **increasing levels of aggressiveness**:

```
Level 0 (structural)  ──parse──► valid? ──render──► ✓ Done
         │ fail
Level 1 (escape chars) ──parse──► valid? ──render──► ✓ Done
         │ fail
Level 2 (strip features)──parse──► valid? ──render──► ✓ Done
         │ fail
Fallback: show source as formatted code block
```

Each level uses `mermaid.parse(text, { suppressErrors: true })` before calling `mermaid.render()` — this validates syntax silently without triggering `console.error` or the Next.js error overlay.

### Sanitization Level Details

#### Level 0 — Structural Fixes (safe for all diagrams)

| Fix | Example |
|-----|---------|
| Convert source citations to comments | `Sources: [file]()` → `%% Source: file` |
| Strip empty markdown links | `[text]()` → `(text)` |
| Auto-declare missing sequence participants | Scans message arrows, adds `participant X` declarations |
| Fix broken arrow tokens | `A )| B` → `A -) B`, `A PS B` → `A -) B` |
| Convert sequence arrows in flowcharts | `A -->> B` → `A -.-> B` |
| Convert colon labels to pipe labels | `A --> B: label` → `A -->|label| B` (flowcharts only) |

#### Level 1 — Escape Special Characters

| Fix | Example |
|-----|---------|
| Curly braces in messages | `Response{Success}` → `Response(Success)` |
| Parentheses in edge labels | `-->|getItem(0)|` → `-->|getItem❨0❩|` |
| Parentheses in bracket labels | `A[func()]` → `A["func❨❩"]` |
| Nested brackets in labels | `A[arr[0]]` → `A["arr⟦0⟧"]` |
| Commas in unquoted labels | `A[x, y]` → `A["x; y"]` |
| Nested parentheses in round nodes | `A(func())` → `A(func❨❩)` |

#### Level 2 — Strip Advanced Features

| Fix | Example |
|-----|---------|
| Remove activation markers | `->>+X:` → `->> X:`, `-->>-X:` → `-->> X:` |
| Force-quote all unsafe labels | `A[complex!label]` → `A["complex!label"]` |

### Rendering Features

- **SVG Pan-Zoom:** When `zoomingEnabled={true}`, diagram renders in a `h-[600px]` container with interactive pan/zoom controls via `svg-pan-zoom` library
- **Fullscreen Modal:** Click-to-zoom opens a modal with zoom in/out/reset controls
- **Dark Mode:** Adds `data-theme="dark"` attribute to SVG for CSS-based theming
- **Unique IDs:** Each `<Mermaid>` component generates a random ID (`mermaid-{random}`) to prevent D3 collisions when multiple diagrams render concurrently

### Error Fallback

When all sanitization levels fail, the diagram source is displayed as a styled code block (dark background, monospace font) — not a red error panel. This provides a readable fallback.

---

## 8. Navigation & Scroll Behavior

### Sidebar Navigation (WikiTreeView)

```
WikiStructure
  ├─ rootSections: ["1", "2", "3", ...]     ← top-level sections
  ├─ sections: [                              ← all section metadata
  │   { id: "1", title: "...", pages: [...], subsections: [...] }
  │   { id: "1.1", title: "...", pages: [...] }
  │ ]
  └─ pages: [                                ← all page metadata
      { id: "1", title: "...", content: "", importance: "high" }
    ]
```

**Rendering rules:**
- Root sections use `renderSection(id, level=0)` — always show chevron
- Inline subsection objects use `renderSectionObj(section, level)` — chevron only if has children
- Section IDs displayed as muted prefix: `1 Project Overview`, `1.1 Repository Layout`
- Expanded children have left border (`border-l`) and indentation (`ml-4 pl-2`)
- Section headers with same ID as an overview page navigate on click; otherwise toggle expand

**Expand state:** `expandedSections: Set<string>` initialized from `rootSections`.

### Page Change Scrolling

```typescript
// Scroll to top when page changes
useEffect(() => {
  const el = document.getElementById('wiki-content');
  if (el) el.scrollTo({ top: 0, behavior: 'smooth' });
}, [currentPageId]);
```

### Deep Linking

URL parameter `?page=pageId` selects a specific page on load.

---

## 9. Chat / Ask System

### Component: `Ask.tsx`

Provides repository Q&A via WebSocket streaming or HTTP fallback.

**Connection routing** (`networkConfig.ts`):
- `localhost`: Direct to `:8001` → `ws://localhost:8001/ws/chat`
- Cloud: nginx proxy → `wss://hostname/ws/chat`

**Features:**
- Streaming responses with `useRef` for synchronous token access
- Deep research mode (multi-step reasoning)
- Conversation history with clear
- Auto-scroll on new content
- Graceful degradation when backend is unavailable

---

## 10. API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/wiki_cache` | GET | Load wiki JSON (tries backend → local file) |
| `/api/wiki/projects` | GET | List all cached wiki projects |
| `/api/lang/config` | GET | Available languages from `lang.json` |
| `/api/models/config` | GET | LLM provider/model list |
| `/api/filters/config` | GET | File filter settings |
| `/api/auth/status` | GET | Authentication status |
| `/api/auth/validate` | POST | Validate Git token |
| `/api/chat/stream` | POST | HTTP streaming chat (fallback) |
| `/api/azure-devops/structure` | GET | Azure DevOps repo structure |

---

## 11. Contexts & Hooks

### LanguageContext (`contexts/LanguageContext.tsx`)

- Auto-detects browser language
- Loads translations from `src/messages/{lang}.json`
- Fallback to English

### WikiGenerationContext (`contexts/WikiGenerationContext.tsx`)

- Tracks background wiki generation progress
- Supports minimize/restore during generation
- Shows completion notification
- Persists state across navigation

### useProcessedProjects (`hooks/useProcessedProjects.ts`)

- Fetches cached projects from `/api/wiki/projects`
- Provides `projects`, `isLoading`, `removeProject()`

---

## 12. Utilities

| File | Functions | Purpose |
|------|-----------|---------|
| `networkConfig.ts` | `getWebSocketUrl()`, `isCloudEnvironment()` | WebSocket URL construction, environment detection |
| `citationProcessor.tsx` | `processCitations()`, `generateFileUrl()` | Replace empty `[file]()` citations with real repo URLs |
| `branchDetection.tsx` | `detectCurrentBranch()` | Auto-detect Git branch from repo info |
| `websocketClient.ts` | `createChatWebSocket()`, `closeWebSocket()` | WebSocket lifecycle management |
| `logger.ts` | `logger.debug/info/warn/error()` | Level-aware logging with deduplication and batching |
| `getRepoUrl.ts` | `getRepoUrl()` | Build repository URL from type + owner + repo |

---

## 13. CSS Architecture

### Design System: Microsoft Fluent UI / Azure

**Color variables** defined in `globals.css` with light/dark mode:

| Variable | Light | Dark | Usage |
|----------|-------|------|-------|
| `--background` | `#f3f2f1` | `#1b1a19` | Page background |
| `--foreground` | `#323130` | `#f3f2f1` | Primary text |
| `--accent-primary` | `#0078d4` | `#2b88d8` | Azure Blue (links, buttons) |
| `--border-color` | `#edebe9` | `#3b3a39` | Borders |
| `--card-bg` | `#ffffff` | `#252423` | Card surfaces |
| `--muted` | `#605e5c` | `#a19f9d` | Secondary text |
| `--highlight` | `#00bcf2` | `#50e6ff` | Azure Cyan |

### Key Classes

| Class | Source | Purpose |
|-------|--------|---------|
| `.btn-azure` | globals.css | Primary button (Azure Blue) |
| `.card-azure` | globals.css | Card with border, overflow-hidden |
| `.shadow-custom` | globals.css | Fluent depth-4 shadow |
| `.scrollbar-thin` | globals.css | Custom 6px scrollbar |
| `.mermaid-diagram` | globals.css | Mermaid node/edge styling with Azure palette |
| `.prose` | Tailwind | Markdown content typography |

### Mermaid Styling

The `.mermaid-diagram` class applies Azure color palette to diagram elements:
- Nodes: `fill: var(--card-bg)`, `stroke: var(--accent-primary)`
- Edges: `stroke: var(--accent-primary)`, `1.5px`
- Text: `fill: var(--muted)` (light mode), `fill: var(--foreground)` (dark mode)
- Dark mode overrides via `html[data-theme='dark'] .mermaid-diagram`

### Body Overflow

```css
body { overflow: hidden; }
```

This prevents a **phantom browser scrollbar** caused by mermaid's `render()` function temporarily injecting full-size SVG elements into `document.body` (outside the React tree) during diagram rendering.

### Responsive Breakpoints

| Breakpoint | Behavior |
|-----------|----------|
| `< lg` (< 1024px) | Sidebar hidden, chat as floating button, stacked layout |
| `≥ lg` (≥ 1024px) | 3-column layout, sidebar visible, chat as right panel |

---

## 14. Type Definitions

```typescript
interface RepoInfo {
  owner: string;
  repo: string;
  type: string;          // 'github' | 'gitlab' | 'bitbucket' | 'azuredevops' | 'local'
  token: string | null;
  branch: string | null;
  localPath: string | null;
  repoUrl: string | null;
}

interface WikiPage {
  id: string;
  title: string;
  content: string;
  filePaths: string[];
  importance: 'high' | 'medium' | 'low';
  relatedPages: string[];
}

interface WikiSection {
  id: string;
  title: string;
  pages: string[];                          // page IDs
  subsections?: WikiSection[] | string[];   // nested or ID references
}

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];                   // top-level section IDs
}
```

---

## 15. Known Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| No virtual scrolling | All page content rendered at once; very large pages may be slow | Pages are typically < 100KB |
| Mermaid concurrent renders | Multiple diagrams render in parallel using D3 + `document.body` | `body { overflow: hidden }` prevents phantom scrollbars |
| WebSocket-only chat | Chat fails silently if backend is down | HTTP streaming fallback via `/api/chat/stream` |
| No error boundaries | Unhandled React errors break the entire page | Mermaid errors caught per-component with fallback |
| LLM diagram quality | Mermaid syntax from LLMs is frequently invalid | 3-level progressive sanitization handles most cases |
| Client-side only | No SSR for wiki content | Fast after initial load; search engines can't index |
