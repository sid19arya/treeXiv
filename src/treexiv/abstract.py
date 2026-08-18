"""Reconstruct plain-text abstracts from OpenAlex's inverted-index representation.

OpenAlex never returns a plain `abstract` string — it returns
`abstract_inverted_index`, a `{word: [position, ...]}` mapping, to avoid
copyright issues with republishing full abstract text verbatim in a
convenient form. We have to invert that back into a string ourselves.
"""

from __future__ import annotations


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild an abstract string from an OpenAlex `abstract_inverted_index`.

    Returns "" for None/empty input (some works genuinely have no abstract
    on record — treated as an empty document downstream, not an error).
    """
    if not inverted_index:
        return ""

    max_position = max(pos for positions in inverted_index.values() for pos in positions)
    slots: list[str] = [""] * (max_position + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots).strip()
