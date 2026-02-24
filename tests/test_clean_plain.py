from utils.html_formatter import clean_plain_text

cases = [
    "**Heading**\nThis is a <b>test</b> paragraph. Second sentence here.",
    "<p>Hello &amp; welcome!</p>\nNew line text without punctuation",
    "**Short** Just a short snippet",
    "",
]

for c in cases:
    print('INPUT:', repr(c))
    out = clean_plain_text(c)
    print('OUTPUT:', out)
    print('LEN:', len(out))
    print('---')
