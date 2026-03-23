"""
Repair truncated wiki structure XML from Azure OpenAI content filter.

When ``finish_reason=content_filter`` truncates a streaming response, we
may have partial XML that is missing closing tags.  This module attempts to
repair the XML so the frontend can still parse a usable wiki structure.

Expected XML format (comprehensive with numbered IDs)::

    <wiki_structure>
      <title>...</title>
      <description>...</description>
      <sections>
        <section id="1">
          <title>...</title>
          <pages><page_ref>1</page_ref></pages>
          <subsections>
            <section id="1.1">
              <title>...</title>
              <pages><page_ref>1.1</page_ref></pages>
            </section>
          </subsections>
        </section>
      </sections>
      <pages>
        <page id="1">
          <title>...</title>
          <description>...</description>
          <importance>...</importance>
          <relevant_files><file_path>...</file_path></relevant_files>
          <related_pages><related>...</related></related_pages>
          <parent_section>1</parent_section>
        </page>
      </pages>
    </wiki_structure>

The concise format omits ``<sections>`` but is otherwise identical.
Supports both numbered IDs (1, 2.1, 2.1.1) and legacy slug IDs (page-1).
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def close_open_tags(xml_text: str) -> str:
    """Close unclosed XML tags by tracking open/close stack.

    Lightweight repair that appends missing closing tags in reverse
    order. Use before ``repair_wiki_structure_xml`` for best results.
    """
    open_stack = []
    tag_re = re.compile(r'<(/?)(\w[\w_]*)(?:\s[^>]*)?\s*/?>')
    for m in tag_re.finditer(xml_text):
        is_closing = m.group(1) == '/'
        tag_name = m.group(2)
        if is_closing:
            for i in range(len(open_stack) - 1, -1, -1):
                if open_stack[i] == tag_name:
                    open_stack.pop(i)
                    break
        else:
            if not m.group(0).endswith('/>'):
                open_stack.append(tag_name)
    for tag in reversed(open_stack):
        xml_text += f'</{tag}>'
    return xml_text


def repair_wiki_structure_xml(partial: str) -> Optional[str]:
    """Attempt to repair truncated wiki structure XML.

    Strategy:
    1. Find the last complete ``</page>`` element.
    2. Truncate after it and close the remaining open tags.
    3. Validate that we extracted at least one complete page.

    Args:
        partial: The partial XML string from the truncated response.

    Returns:
        Repaired XML string if repair was possible, else ``None``.
    """
    if not partial:
        return None

    # Strip any leading non-XML content (reasoning model may emit
    # thinking tokens before the XML even with /no_think).
    ws_start = partial.find('<wiki_structure>')
    if ws_start < 0:
        # Maybe partial starts with <wiki_structure but was cut inside
        ws_start = partial.find('<wiki_structure')
        if ws_start < 0:
            logger.warning("No <wiki_structure> tag found in partial XML")
            return None

    working = partial[ws_start:]

    # If the XML is already complete, return as-is
    if '</wiki_structure>' in working:
        logger.info("Partial XML already contains </wiki_structure>")
        return working

    # ------------------------------------------------------------------
    # Strategy A: Truncate after the last complete </page> element
    # ------------------------------------------------------------------
    last_page_end = working.rfind('</page>')
    if last_page_end > 0:
        # Include the closing tag itself
        truncated = working[:last_page_end + len('</page>')]

        # Determine which containing tags are still open.
        # If we're inside <pages>...</pages>, close it.
        # Then close <wiki_structure>.
        needs_pages_close = (
            '<pages>' in truncated
            and '</pages>' not in truncated.split('</page>')[-1]
        )
        suffix = ''
        if needs_pages_close:
            suffix += '\n  </pages>'
        suffix += '\n</wiki_structure>'

        repaired = truncated + suffix

        # Count how many complete pages we captured
        page_count = len(re.findall(r'</page>', repaired))
        logger.info(
            "Repaired wiki structure XML: %d complete pages "
            "(truncated after last </page>)",
            page_count,
        )
        return repaired

    # ------------------------------------------------------------------
    # Strategy B: No complete <page> elements, but we may have sections
    # ------------------------------------------------------------------
    # Try to find complete sections at least, and create stub pages
    last_section_end = working.rfind('</section>')
    if last_section_end > 0:
        truncated = working[:last_section_end + len('</section>')]

        # Extract page_ref IDs from sections (numbered or slug format)
        page_refs = re.findall(
            r'<page_ref>([\w.-]+)</page_ref>', truncated
        )

        # Close sections tag if open
        needs_sections_close = (
            '<sections>' in truncated
            and '</sections>' not in truncated
        )
        suffix = ''
        if needs_sections_close:
            suffix += '\n  </sections>'

        # Generate stub pages from page_refs
        if page_refs:
            suffix += '\n  <pages>'
            for ref_id in page_refs:
                suffix += (
                    f'\n    <page id="{ref_id}">'
                    f'<title>Section Page</title>'
                    f'<description>Page content will be generated from '
                    f'source code.</description>'
                    f'<importance>medium</importance>'
                    f'<relevant_files></relevant_files>'
                    f'<related_pages></related_pages>'
                    f'</page>'
                )
            suffix += '\n  </pages>'

        suffix += '\n</wiki_structure>'
        repaired = truncated + suffix

        logger.info(
            "Repaired wiki structure XML from sections: "
            "%d stub pages from page_refs",
            len(page_refs),
        )
        return repaired

    # ------------------------------------------------------------------
    # Strategy C: Minimal content — only title/description available
    # ------------------------------------------------------------------
    # Extract title and description, create a minimal structure
    title_match = re.search(
        r'<title>(.*?)</title>', working, re.DOTALL
    )
    desc_match = re.search(
        r'<description>(.*?)</description>', working, re.DOTALL
    )
    if title_match:
        title = title_match.group(1).strip()
        description = desc_match.group(1).strip() if desc_match else ''
        repaired = (
            f'<wiki_structure>\n'
            f'  <title>{title}</title>\n'
            f'  <description>{description}</description>\n'
            f'  <pages>\n'
            f'    <page id="page-1">'
            f'<title>Overview</title>'
            f'<description>General overview of the repository.</description>'
            f'<importance>high</importance>'
            f'<relevant_files></relevant_files>'
            f'<related_pages></related_pages>'
            f'</page>\n'
            f'    <page id="page-2">'
            f'<title>Architecture</title>'
            f'<description>System architecture and design.</description>'
            f'<importance>high</importance>'
            f'<relevant_files></relevant_files>'
            f'<related_pages></related_pages>'
            f'</page>\n'
            f'    <page id="page-3">'
            f'<title>Core Components</title>'
            f'<description>Key components and modules.</description>'
            f'<importance>medium</importance>'
            f'<relevant_files></relevant_files>'
            f'<related_pages></related_pages>'
            f'</page>\n'
            f'  </pages>\n'
            f'</wiki_structure>'
        )
        logger.info(
            "Repaired wiki structure XML with minimal fallback "
            "(only title/description found)"
        )
        return repaired

    logger.warning("Could not repair partial wiki structure XML")
    return None
