"""Minimal local fallback for environments without the external ftfy package.

The original project uses ftfy.fix_text during tokenization cleanup. For offline
inference, returning the input text unchanged keeps the pipeline functional.
"""


def fix_text(text):
    return text
