import re
import html
from typing import List


def format_description_to_html(text: str) -> str:
    """Convert markdown-style sections into Shopify-safe HTML.

    Rules applied:
    - Lines that are a bold-only heading like "**Heading**" become <h2>Heading</h2>
    - Contiguous non-heading lines form a paragraph wrapped in <p>..</p>
    - Single newlines inside a paragraph are preserved as <br/>
    - Blank lines separate paragraphs
    - HTML is escaped for safety

    Returns a clean HTML string (blocks separated by newlines).
    """
    if not text:
        return ""

    heading_re = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
    lines: List[str] = text.splitlines()

    blocks: List[str] = []
    para_lines: List[str] = []

    def flush_paragraph():
        nonlocal para_lines
        if para_lines:
            # escape each line and preserve line breaks using <br/>
            escaped = [html.escape(l) for l in para_lines]
            blocks.append("<p>" + "<br/>".join(escaped) + "</p>")
            para_lines = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            # flush any open paragraph first
            flush_paragraph()
            heading = m.group(1).strip()
            blocks.append(f"<h2>{html.escape(heading)}</h2>")
            continue

        # blank line -> paragraph boundary
        if line.strip() == "":
            flush_paragraph()
            continue

        # regular paragraph line
        para_lines.append(line)

    # flush final paragraph
    flush_paragraph()

    # join blocks with single newlines
    return "\n".join(blocks)


def clean_plain_text(text: str) -> str:
    """Return a single-line, SEO-friendly sentence from raw text.

    Steps:
    - remove markdown bold markers (`**`)
    - strip HTML tags and unescape entities
    - collapse whitespace and line breaks
    - return the first sentence (or the whole text if none)
    - truncate to 155 characters without cutting the last word
    """
    if not text:
        return ""

    # remove bold markers
    s = text.replace("**", " ")

    # unescape HTML entities
    s = html.unescape(s)

    # remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)

    # normalize whitespace and remove line breaks
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # attempt to extract the first sentence (up to first .!?)
    m = re.search(r"(.+?[\.\!?])\s", s + " ")
    if m:
        sentence = m.group(1).strip()
    else:
        sentence = s

    # enforce max length 155 chars without cutting words
    max_len = 155
    if len(sentence) > max_len:
        trunc = sentence[:max_len]
        if " " in trunc:
            trunc = trunc.rsplit(" ", 1)[0]
        sentence = trunc.rstrip(" ,;:-")

    return sentence
