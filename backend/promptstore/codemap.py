"""
CodeMap prompt templates and builders.

Templates:
    CODEMAP_SECTION_TEMPLATE — Numbered section for wiki structure prompt injection

Builders:
    build_codemap_prompt_section() — Format codemap summary as a prompt section
"""


CODEMAP_SECTION_TEMPLATE = (
    '3. Code architecture analysis (AST-based) showing modules, '
    'key symbols, and dependencies:\n'
    '{codemap_summary}\n\n'
    'Use the codemap above to organize wiki pages around actual '
    'architectural boundaries (modules, key classes, dependency '
    'clusters) rather than just directory layout.\n'
)


def build_codemap_prompt_section(codemap_summary: str) -> str:
    """Format a codemap summary as a numbered prompt section.

    Returns an empty string when no summary is available, so the prompt
    template's {additional_context} placeholder collapses cleanly.

    Args:
        codemap_summary: Compact codemap summary text from
            codemap_generator.summarize_codemap()

    Returns:
        Formatted prompt section string, or '' if no summary
    """
    if not codemap_summary:
        return ''
    return CODEMAP_SECTION_TEMPLATE.format(
        codemap_summary=codemap_summary,
    )
