import re
from typing import List, Dict, Any
from src.config import settings
from src.utils.logging_utils import get_logger

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

def simple_text_chunker(text: str, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into chunks with overlap, suitable for general text or code.

    Args:
        text: The input text to be chunked.
        chunk_size: The target size for each chunk.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_overlap >= chunk_size:
        logger.warning(f"Chunk overlap ({chunk_overlap}) is >= chunk size ({chunk_size}). Setting overlap to {chunk_size // 3}")
        chunk_overlap = chunk_size // 3

    chunks = []
    start = 0
    text_length = len(text)
    if text_length == 0:
        return []
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        
        if end == text_length:
            break
        
        start += (chunk_size - chunk_overlap)
        if start >= end:
            start = end
            
    return [chunk for chunk in chunks if chunk and chunk.strip()]


def smart_chunk_markdown(text: str, chunk_size: int = settings.MARKDOWN_CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks, respecting code blocks and paragraphs.

    Args:
        text: The input text to be chunked.
        chunk_size: The target maximum size for each chunk.

    Returns:
        A list of text chunks.
    """
    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + chunk_size, text_len)
        current_chunk_text = text[current_pos:end_pos]

        if end_pos == text_len:
            chunks.append(current_chunk_text.strip())
            break

        para_break = current_chunk_text.rfind('\\n\\n')
        code_fence_break = current_chunk_text.rfind('\\n```')
        sentence_end_match = list(re.finditer(r'[.!?](\\s|\\n)+', current_chunk_text))
        sentence_break = sentence_end_match[-1].start() + 1 if sentence_end_match else -1
        
        split_at = -1
        if para_break > chunk_size * 0.5:
            split_at = para_break + 2
        elif code_fence_break > chunk_size * 0.3:
            if text[current_pos+code_fence_break:].startswith('```\\n'):
                 split_at = code_fence_break + 4
            else:
                 split_at = code_fence_break + 3
        elif sentence_break > chunk_size * 0.4:
            split_at = sentence_break
        
        if split_at != -1 and (current_pos + split_at) < end_pos :
            final_chunk = text[current_pos : current_pos + split_at].strip()
            current_pos += split_at
        else:
            final_chunk = text[current_pos:end_pos].strip()
            current_pos = end_pos
        
        if final_chunk:
            chunks.append(final_chunk)

    return [c for c in chunks if c]
