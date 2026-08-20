"""Guessing which spreadsheet column plays which role.

The guess is never final. It pre-fills the dropdowns on screen so a person
confirms it, because a wrong mapping is invisible in the result.
"""
import difflib
import re
from typing import Dict, NamedTuple, Optional, Sequence

ROLES = ("task", "optimistic", "realistic", "pessimistic")
REQUIRED_ROLES = ("task", "optimistic", "realistic", "pessimistic")

SYNONYMS = {
    "task": [
        "task", "task name", "name", "activity", "work item", "work package",
        "workpackage", "description", "deliverable", "wbs", "item", "step",
        "задача", "работа", "этап", "наименование", "название",
    ],
    "optimistic": [
        "optimistic", "optimistic days", "best", "best case", "best case d",
        "min", "minimum", "low", "lo", "shortest", "o",
        "оптимистичная", "оптимистичный", "оптимистично", "минимум", "лучший",
    ],
    "realistic": [
        "realistic", "realistic days", "most likely", "mostlikely", "likely",
        "expected", "expectation", "estimate", "mode", "normal", "ml", "m",
        "real",
        "реалистичная", "реалистичный", "ожидаемая", "наиболее вероятная",
        "вероятная", "оценка",
    ],
    "pessimistic": [
        "pessimistic", "pessimistic days", "worst", "worst case",
        "worst case d", "max", "maximum", "high", "hi", "longest", "p",
        "пессимистичная", "пессимистичный", "максимум", "худший",
    ],
}

FUZZY_CUTOFF = 0.84
MIN_FUZZY_LENGTH = 4  # short synonyms like "o" or "max" fuzzy-match anything


class RoleGuess(NamedTuple):
    """One role's best candidate column and how it was found."""

    column: Optional[str]
    confidence: str  # "exact", "fuzzy" or "none"


def normalize_header(raw: str) -> str:
    """Reduce a header to lowercase words separated by single spaces."""
    text = str(raw).lower().strip()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _exact_pass(
    normalized: Dict[str, str], taken: set
) -> Dict[str, RoleGuess]:
    """Match headers whose normalised form is a synonym outright."""
    found = {}
    for role in ROLES:
        vocabulary = set(SYNONYMS[role])
        for column, norm in normalized.items():
            if column in taken:
                continue
            if norm in vocabulary:
                found[role] = RoleGuess(column, "exact")
                taken.add(column)
                break
    return found


def _fuzzy_pass(
    normalized: Dict[str, str], taken: set, unresolved: Sequence[str]
) -> Dict[str, RoleGuess]:
    """Recover typos and suffixes for roles the exact pass missed."""
    found = {}
    for role in unresolved:
        vocabulary = [s for s in SYNONYMS[role] if len(s) >= MIN_FUZZY_LENGTH]
        best_column = None
        best_score = 0.0
        for column, norm in normalized.items():
            if column in taken:
                continue
            for word in [norm] + norm.split():
                matches = difflib.get_close_matches(
                    word, vocabulary, n=1, cutoff=FUZZY_CUTOFF
                )
                if not matches:
                    continue
                score = difflib.SequenceMatcher(None, word, matches[0]).ratio()
                if score > best_score:
                    best_score, best_column = score, column
        if best_column is not None:
            found[role] = RoleGuess(best_column, "fuzzy")
            taken.add(best_column)
    return found


def guess_mapping(columns: Sequence[str]) -> Dict[str, RoleGuess]:
    """Propose a column for every role.

    Two passes: exact synonym match first, then fuzzy matching for whatever
    is left. A column is claimed by at most one role.

    Args:
        columns: the sheet's column names, as written.

    Returns:
        A guess for every role in ROLES; unmatched roles get
        ``RoleGuess(None, "none")``.
    """
    normalized = {column: normalize_header(column) for column in columns}
    taken = set()

    guesses = _exact_pass(normalized, taken)
    unresolved = [role for role in ROLES if role not in guesses]
    guesses.update(_fuzzy_pass(normalized, taken, unresolved))

    return {role: guesses.get(role, RoleGuess(None, "none")) for role in ROLES}


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return str(value).strip() == ""


def _row_score(cells: Sequence) -> int:
    """How strongly a row of raw cell values reads as column headers."""
    texts = [str(v) for v in cells if not _is_blank(v)]
    if not texts:
        return 0
    guess = guess_mapping(texts)
    weights = {"exact": 2, "fuzzy": 1, "none": 0}
    return sum(weights[guess[role].confidence] for role in REQUIRED_ROLES)


def detect_header_row(rows: Sequence[Sequence]) -> int:
    """Guess which row of a headerless preview actually holds column titles.

    Spreadsheets often carry a title or a merged section header above the
    real table (a two-row header like "Back-End" spanning "min / real /
    max" is a common case). Each row is scored by how many required roles
    it matches; the best-scoring row wins. Ties keep the earliest row.

    Args:
        rows: raw rows, each a sequence of cell values, no header assumed.

    Returns:
        The zero-based index of the best-scoring row, or 0 if nothing in
        the preview scores above zero (matching the previous default of
        treating the first row as the header).
    """
    best_index = 0
    best_score = 0
    for index, row in enumerate(rows):
        score = _row_score(row)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index
