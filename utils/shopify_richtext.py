import re
from typing import Dict, List, Any


def to_shopify_richtext(text: str) -> Dict[str, Any]:
    """Convert plain text into Shopify Rich Text metafield JSON using intelligent parsing.

    Rules implemented:
    - Split text into blocks using double line breaks (`"\n\n"`).
    - Within each block, detect merged headings using the provided regex
      and split them from following sentence text.
    - A heading is considered to be a short phrase that:
      - starts with a capital letter
      - contains between 2 and 8 words
      - does not end with a period
      - is immediately followed by sentence text (this is detected by the split)

    The regex used to split merged headings from paragraphs is:
    r'(?<=[a-z])(?=[A-Z][a-z]+(?:\\s[A-Z][a-z]+){0,6})'

    Returns a Shopify RichText-compatible dict with `type: root` and
    children consisting of heading and paragraph nodes.
    """
    root: Dict[str, Any] = {"type": "root", "children": []}
    if not text:
        return root

    # Normalize line endings
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split input into blocks by double line breaks
    raw_blocks = s.split("\n\n") if s else []
    blocks: List[str] = [b.strip() for b in raw_blocks if b and b.strip()]

    # Regex to split merged heading from paragraph
    split_pattern = re.compile(r'(?<=[a-z])(?=[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,6})')

    children: List[Dict[str, Any]] = []

    # Heading validation regex: 2-8 words, each starting with capital letter
    heading_validate = re.compile(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,7}$')

    for block in blocks:
        # Apply the split pattern to separate any merged heading from following text
        parts = split_pattern.split(block)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If part looks like a heading according to validation rules and
            # does not end with terminal punctuation, treat it as heading
            is_heading = bool(heading_validate.match(part)) and not re.search(r"[\.\!\?]$", part)

            if is_heading:
                heading_node = {"type": "heading", "level": 2, "children": [{"type": "text", "value": part}]}
                children.append(heading_node)
            else:
                paragraph_node = {"type": "paragraph", "children": [{"type": "text", "value": part}]}
                children.append(paragraph_node)

    root["children"] = children
    return root
