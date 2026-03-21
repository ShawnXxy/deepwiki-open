"""
Wiki page content template and builder.

Template:
    WIKI_PAGE_CONTENT_PROMPT — Instructions for LLM to generate one wiki page

Builders:
    build_wiki_page_prompt()   — Assembles full prompt with context, files, cross-refs
    format_file_paths_list()   — Formats file paths as commit-pinned markdown links
    format_page_catalog()      — Formats page list for LLM cross-referencing
"""

WIKI_PAGE_CONTENT_PROMPT = """You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about "{page_title}" within the given software project.

NOTE: When describing code, focus on architecture, design patterns, data flow,
and component relationships. Summarize and explain code in your own words rather
than quoting large blocks of raw source verbatim. Avoid reproducing credentials,
secrets, security rules, or sensitive configuration values.

CONTENT SAFETY: If source code contains informal, slang, or potentially
offensive terms in identifiers, comments, or test names, do NOT reproduce
them verbatim. Describe their purpose using professional language instead.
This applies to variable names, function names, file paths, and comments.

You will be given:
1. The wiki page topic: "{page_title}"
2. A list of relevant source files from the project that you should use as the basis for the content.

INSTRUCTIONS:
- Generate the wiki content based on the provided files, even if there are only 1-2 files.
- Focus on the information available in the source files.
- Work with whatever source files are provided.

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a `<details>` block listing ALL the relevant source files you used to generate the content.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

{file_paths_list}
</details>

Immediately after the `<details>` block, the main title of the page should be a H1 Markdown heading: `# {page_title}`.

UNDERSTANDING THE SOURCE CONTEXT:
The source files provided may include structural metadata to help you write more accurate content:
- **File headers** like `## File Path: path/to/file.py` indicate the source file.
- **Structural summaries** like `(Type: py | Classes: MyClass | Functions: init, process)` list key components defined in the file.
- **Section markers** like `### [function]` or `### [class]` indicate what type of code block follows.
- **Line references** like `(lines 45-120)` indicate where in the source file the code appears.
- **Source URLs** like `Source: [file.py L45-L120](https://...)` provide commit-pinned permalink URLs. COPY THESE EXACTLY into your citations.
Use this structural information to write more precise descriptions and cite specific components accurately.

Based on the content of the relevant source files:

1.  **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "{page_title}" within the context of the overall project.

2.  **Detailed Sections:** Break down "{page_title}" into logical sections using H2 (`##`) and H3 (`###`) Markdown headings. For each section:
    *   Explain the architecture, components, data flow, or logic relevant to the section's focus, as evidenced in the source files.
    *   Identify key functions, classes, data structures, API endpoints, or configuration elements pertinent to that section.

3.  **Mermaid Diagrams:**
    *   Use Mermaid diagrams (e.g., `flowchart TD`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `graph TD`) to visually represent architectures, flows, relationships, and schemas found in the source files.
    *   Ensure diagrams are accurate and directly derived from the source files.
    *   Provide a brief explanation before or after each diagram to give context.
    *   CRITICAL: All diagrams MUST follow strict vertical orientation:
       - Use "graph TD" (top-down) directive for flow diagrams
       - NEVER use "graph LR" (left-right)
       - Maximum node width should be 3-4 words
       - For sequence diagrams:
         - Start with "sequenceDiagram" directive on its own line
         - Define ALL participants at the beginning using "participant" keyword
         - Optionally specify participant types: actor, boundary, control, entity, database, collections, queue
         - Use descriptive but concise participant names, or use aliases: "participant A as Alice"
         - Use the correct Mermaid arrow syntax (8 types available):
           - -> solid line without arrow (rarely used)
           - --> dotted line without arrow (rarely used)
           - ->> solid line with arrowhead (most common for requests/calls)
           - -->> dotted line with arrowhead (most common for responses/returns)
           - ->x solid line with X at end (failed/error message)
           - -->x dotted line with X at end (failed/error response)
           - -) solid line with open arrow (async message, fire-and-forget)
           - --) dotted line with open arrow (async response)
           - Examples: A->>B: Request, B-->>A: Response, A->xB: Error, A-)B: Async event
         - Use +/- suffix for activation boxes: A->>+B: Start (activates B), B-->>-A: End (deactivates B)
         - Group related participants using "box": box GroupName ... end
         - Use structural elements for complex flows:
           - loop LoopText ... end (for iterations)
           - alt ConditionText ... else ... end (for conditionals)
           - opt OptionalText ... end (for optional flows)
           - par ParallelText ... and ... end (for parallel actions)
           - critical CriticalText ... option ... end (for critical regions)
           - break BreakText ... end (for breaking flows/exceptions)
         - Add notes for clarification: "Note over A,B: Description", "Note right of A: Detail"
         - Use autonumber directive to add sequence numbers to messages
         - NEVER use flowchart-style labels like A--|label|-->B. Always use a colon for labels: A->>B: My Label
         - CRITICAL: ALWAYS declare participants before using them. Using undeclared participants will cause parse errors.
         - CRITICAL: Participant names must be simple identifiers (letters, numbers, underscores). Avoid special characters.
         - CRITICAL: Use activate/deactivate correctly: send message with + suffix, return with - suffix
         - CRITICAL: Test your diagram mentally - ensure every participant used is declared at the top

4.  **Tables:**
    *   Use Markdown tables to summarize information such as:
        *   Key features or components and their descriptions.
        *   API endpoint parameters, types, and descriptions.
        *   Configuration options, their types, and default values.
        *   Data model fields, types, constraints, and descriptions.

5.  **Code Snippets (OPTIONAL):**
    *   Include short, relevant code snippets (e.g., Python, Java, JavaScript, SQL, JSON, YAML) directly from the relevant source files to illustrate key implementation details, data structures, or configurations.
    *   Ensure snippets are well-formatted within Markdown code blocks with appropriate language identifiers.

6.  **Source Citations:**
    *   Each code chunk in the SOURCE CODE CONTEXT may include a `Source:` line with a permalink URL.
    *   COPY THE EXACT URL from these `Source:` lines when citing. Do not modify or fabricate URLs.
    *   Place citations at the end of each major section (after each H2 or H3).
    *   Use the format: `Sources: [filename.ext L10-L25](url)`.
    *   Multiple citations on one line: `Sources: [a.py L10-L25](url1), [b.py L30-L45](url2)`.
    *   If no `Source:` URL is available for a chunk, use an empty URL: `Sources: [filename.ext]()`.
    *   When structural metadata lists specific functions or classes, reference them by name in your explanations.
7.  **Technical Accuracy:** Base all information on the provided source files. If information is limited, focus on what IS available rather than what's missing.

8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project.

9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page_title}".

IMPORTANT: Generate the content in {language_name} language.

REMINDERS:
- Generate content based on available source files.
- Focus on the information available, not what might be missing.
- NEVER ask clarifying questions or request additional files.
- If the provided context is limited, write about the topic using file names, directory structure, and any metadata available.
"""


def format_file_paths_list(
    file_paths: list,
    repo_url: str = "",
    branch: str = "main",
    commit_hash: str = None,
    repo_type: str = "github",
) -> str:
    """
    Format a list of file paths as markdown links for the wiki page header.
    Uses commit-pinned URLs when commit_hash is provided.
    """
    if not file_paths:
        return "- No source files specified"

    if repo_url and commit_hash:
        from backend.utils.url_builder import build_source_url
        links = []
        for path in file_paths:
            url = build_source_url(
                repo_url, path, commit_hash, repo_type
            )
            links.append(f"- [{path}]({url})")
        return "\n".join(links)
    elif repo_url:
        return "\n".join([
            f"- [{path}]({repo_url}?path=/{path}&version=GB{branch})"
            for path in file_paths
        ])
    else:
        return "\n".join([f"- {path}" for path in file_paths])


def format_page_catalog(
    pages: list, current_page_id: str = ""
) -> str:
    """
    Format a list of wiki pages for cross-referencing.

    Args:
        pages: List of dicts with 'id' and 'title' keys
        current_page_id: ID of the current page (excluded)

    Returns:
        Formatted catalog string for prompt injection
    """
    lines = []
    for page in pages:
        pid = page.get('id', '') if isinstance(page, dict) else getattr(page, 'id', '')
        title = page.get('title', '') if isinstance(page, dict) else getattr(page, 'title', '')
        if pid and pid != current_page_id:
            lines.append(f"- {pid}: {title}")
    return "\n".join(lines)


def build_wiki_page_prompt(
    page_title: str,
    page_id: str,
    file_paths: list,
    context_text: str,
    repo_url: str = "",
    commit_hash: str = None,
    page_catalog: str = None,
    file_summaries: str = None,
    language_name: str = "English",
    repo_type: str = "github",
) -> str:
    """
    Build the complete wiki page generation prompt.

    Single source of truth — replaces the frontend template.
    Injects server-side data (commit hash, page catalog,
    file summaries) that the frontend can't access.
    """
    file_paths_list = format_file_paths_list(
        file_paths, repo_url, commit_hash=commit_hash,
        repo_type=repo_type,
    )

    prompt = WIKI_PAGE_CONTENT_PROMPT.format(
        page_title=page_title,
        file_paths_list=file_paths_list,
        language_name=language_name,
    )

    # Cross-page references
    if page_catalog:
        prompt += (
            "\n\nOTHER WIKI PAGES (for cross-referencing):\n"
            "When relevant, create inline links using the format "
            "[Page Title](deepwiki://page_id).\n"
            "Example: \"For details, see "
            "[Dependency Injection](deepwiki://2.2).\"\n\n"
            f"{page_catalog}\n"
        )

    # File summaries for architectural context
    if file_summaries:
        prompt += f"\n\nFILE SUMMARIES:\n{file_summaries}\n"

    # Source code context from RAG
    if context_text and context_text.strip():
        prompt += (
            f"\n\nSOURCE CODE CONTEXT:\n{context_text}\n"
        )

    return prompt
