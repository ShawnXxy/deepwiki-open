"""
Wiki Structure Generation prompts.

These prompts are used to analyze a repository and generate the wiki structure
(table of contents) with sections and pages.

Note: These prompts use placeholders that need to be filled in:
- {owner}: Repository owner
- {repo}: Repository name
- {file_tree}: The complete file tree
- {readme}: The README content
- {language_name}: Target language for wiki content
- {page_count}: Number of pages to generate (e.g., "8-12" or "4-6")
"""

# Comprehensive wiki structure prompt (includes sections)
WIKI_STRUCTURE_PROMPT = """Analyze this repository {owner}/{repo} and create a wiki structure for it.

NOTE: The file tree and README below are provided as structural context only.
Focus on file names, directory structure, and high-level descriptions.
Do NOT reproduce or quote raw code, credentials, security rules,
or sensitive content from the repository. Your task is purely to
design a wiki table of contents.

CONTENT SAFETY: When creating page titles and descriptions, use clean
professional language only. Do NOT reproduce file or variable names that
contain informal, slang, or potentially offensive terms. Paraphrase them
with professional descriptions instead.

1. The complete file tree of the project:
<file_tree>
{file_tree}
</file_tree>

2. The README file of the project:
<readme>
{readme}
</readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in {language_name} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

Create a structured wiki with UP TO 3 LEVELS of hierarchy using the following categories as guidance (include only those relevant to this repository):
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality, broken into sub-topics)
- Data Management/Flow (database schema, data pipelines, state management)
- Frontend Components (UI elements, if applicable)
- Backend Systems (server-side components)
- Model Integration (AI model connections, if applicable)
- Testing and Quality Assurance (test frameworks, coverage, CI checks)
- Deployment/Infrastructure (how to deploy, CI/CD pipeline, containerization)
- Development Workflow (contributing, local setup, code style)
- Extensibility and Customization (plugins, theming, custom modules, hooks)

NUMBERING SYSTEM: Use numbered IDs that encode position in the hierarchy:
- Top-level sections: 1, 2, 3, ...
- Sub-sections: 2.1, 2.2, 2.3, ...
- Pages within a section get the section's number: section 2 contains page "2", section 2.1 contains page "2.1"

Each section MUST contain one or more pages. EVERY page you define MUST be assigned to exactly one section via <page_ref>.

Sections can contain <subsections> with nested <section> elements for 3-level depth. Example:
  <section id="2">
    <title>Core Framework</title>
    <pages><page_ref>2</page_ref></pages>
    <subsections>
      <section id="2.1">
        <title>Routing System</title>
        <pages><page_ref>2.1</page_ref></pages>
      </section>
      <section id="2.2">
        <title>Middleware</title>
        <pages><page_ref>2.2</page_ref></pages>
      </section>
    </subsections>
  </section>

Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="1">
      <title>[Section title]</title>
      <pages>
        <page_ref>1</page_ref>
      </pages>
      <subsections>
        <section id="1.1">
          <title>[Subsection title]</title>
          <pages><page_ref>1.1</page_ref></pages>
        </section>
      </subsections>
    </section>
    <!-- More sections as needed -->
  </sections>
  <pages>
    <page id="1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>2.1</related>
        <!-- More related page IDs as needed -->
      </related_pages>
      <parent_section>1</parent_section>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in markdown code blocks (no ``` or ```xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

CRITICAL VALIDATION RULES:
1. Create {page_count} pages that would make a comprehensive wiki for this repository
2. Use numbered IDs (1, 2.1, 2.1.1) that match position in the hierarchy
3. Each page should focus on a specific aspect of the codebase
4. The relevant_files should be actual files from the repository that would be used to generate that page
5. **EVERY page MUST be assigned to exactly one section** - each page's id MUST appear as a <page_ref> in one of the <section> elements
6. Verify before returning: count of <page> elements MUST equal count of <page_ref> elements across all sections
7. Do NOT create a generic "Additional Topics" or "Miscellaneous" section. Every page must belong to a meaningful, descriptive section
8. Do NOT generate duplicate pages covering the same topic under different IDs
9. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters"""

# Concise wiki structure prompt (no sections, simpler structure)
WIKI_STRUCTURE_CONCISE_PROMPT = """Analyze this GitHub repository {owner}/{repo} and create a wiki structure for it.

NOTE: The file tree and README below are provided as structural context only.
Focus on file names, directory structure, and high-level descriptions.
Do NOT reproduce or quote raw code, credentials, security rules,
or sensitive content from the repository. Your task is purely to
design a wiki table of contents.
CONTENT SAFETY: When creating page titles and descriptions, use clean
professional language only. Do NOT reproduce file or variable names that
contain informal, slang, or potentially offensive terms. Paraphrase them
with professional descriptions instead.
1. The complete file tree of the project:
<file_tree>
{file_tree}
</file_tree>

2. The README file of the project:
<readme>
{readme}
</readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in {language_name} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in markdown code blocks (no ``` or ```xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

IMPORTANT:
1. Create {page_count} pages that would make a concise wiki for this repository
2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the repository that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters"""


def get_language_name(language_code: str) -> str:
    """Convert language code to human-readable language name."""
    language_map = {
        'en': 'English',
        'ja': 'Japanese (日本語)',
        'zh': 'Mandarin Chinese (中文)',
        'zh-tw': 'Traditional Chinese (繁體中文)',
        'es': 'Spanish (Español)',
        'kr': 'Korean (한国語)',
        'vi': 'Vietnamese (Tiếng Việt)',
        'pt-br': 'Brazilian Portuguese (Português Brasileiro)',
        'fr': 'Français (French)',
        'ru': 'Русский (Russian)',
    }
    return language_map.get(language_code, 'English')
