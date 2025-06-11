import re
from typing import List, Dict, Any
from ..config import settings
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def extract_code_blocks(text_content: str, min_length: int = settings.CODE_BLOCK_MIN_LENGTH) -> List[Dict[str, Any]]:
    """
    Extracts fenced code blocks from text content (primarily aimed at Markdown).

    Args:
        text_content: The text content to parse.
        min_length: The minimum character length for a code block to be extracted.
                       Defaults to settings.CODE_BLOCK_MIN_LENGTH.

    Returns:
        A list of dictionaries, where each dictionary represents a code block
        and contains 'code' and 'language' (if specified).
    """
    code_blocks = []
    # Regex to find fenced code blocks, capturing optional language
    # ```python (language)
    # code block content
    # ```
    # Handles cases with or without language specifier, and various fence lengths (```, ````, etc.)
    # Non-greedy content match (.*?) to handle multiple blocks correctly.
    # Anchors ^ and $ with re.MULTILINE ensure each block is processed on its own lines.
    pattern = re.compile(r"^(?P<fence>[`~]{3,})\s*(?P<language>\w+)?\s*\n(?P<code>.*?)\n(?P=fence)\s*$", re.MULTILINE | re.DOTALL)

    for match in pattern.finditer(text_content):
        code = match.group('code').strip()
        language = match.group('language') if match.group('language') else None

        if len(code) >= min_length:
            code_blocks.append({
                "code": code,
                "language": language,
                "original_text_length": len(text_content) # For context, if needed later
            })
            logger.debug(f"Extracted code block (lang: {language}, len: {len(code)})")
        else:
            logger.debug(f"Skipped code block (lang: {language}, len: {len(code)}) due to min_length ({min_length}) requirement.")
    return code_blocks
