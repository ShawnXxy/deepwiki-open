"""
Content sanitization for Azure OpenAI content-filter avoidance.

Strips patterns from text that commonly trigger Azure OpenAI's
profanity / PII-detection content-filter layer (the blocklist layer
that fires independently of the severity-based hate/violence/sexual
categories).

Known triggers (discovered via diagnostic log analysis):
- Azure subscription / resource GUIDs
- ``az login`` / ``az account`` CLI commands
- Internal Microsoft ADO / VSTS URLs
- Credential assignments (password=, secret=, api_key=, ...)
- Connection strings (Server=...; ...)

Used by:
- Wiki generation (README + RAG context)
- Chat service (RAG context)
"""

import re


def sanitize_for_content_filter(text: str) -> str:
    """Strip patterns that commonly trigger Azure OpenAI's content filter.

    Returns the text with known trigger patterns replaced by safe
    placeholders.  Designed to have zero impact on content quality —
    the replaced items (GUIDs, login commands, internal URLs,
    credentials) are irrelevant to LLM understanding of code
    architecture.
    """
    # GUIDs (subscription IDs, resource IDs, tenant IDs)
    text = re.sub(
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '<GUID_REDACTED>',
        text,
    )
    # az CLI login / account commands (full lines)
    text = re.sub(
        r'^.*\baz\s+(?:login|account)\b.*$',
        '<AZ_CLI_COMMAND_REDACTED>',
        text, flags=re.MULTILINE | re.IGNORECASE,
    )
    # Internal Microsoft ADO / VSTS / dev.azure.com URLs
    text = re.sub(
        r'https?://[^\s)\]]*(?:visualstudio\.com|dev\.azure\.com)[^\s)\]]*',
        '<INTERNAL_URL_REDACTED>',
        text,
    )
    # Password / secret / key assignments
    text = re.sub(
        r'(["\']?(?:password|secret|api_key|token|credential|auth_token'
        r'|private_key|client_secret)["\']?\s*[:=]\s*)["\'][^"\']{4,}["\']',
        r'\1"<REDACTED>"',
        text, flags=re.IGNORECASE,
    )
    # Connection strings
    text = re.sub(
        r'(?:Server|Data Source|Host)=[^;\n]{10,}',
        '<CONNECTION_STRING_REDACTED>',
        text, flags=re.IGNORECASE,
    )
    return text
