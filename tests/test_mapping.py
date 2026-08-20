"""Tests for turning spreadsheet headers into roles."""
from montecarlo.core.mapping import (
    REQUIRED_ROLES,
    ROLES,
    detect_header_row,
    guess_mapping,
    normalize_header,
)


def test_normalisation_lowercases_and_strips_punctuation():
    assert normalize_header("  Best Case (d) ") == "best case d"
    assert normalize_header("Most-Likely") == "most likely"


def test_canonical_english_headers_map_exactly():
    guess = guess_mapping(["Task", "Optimistic", "Realistic", "Pessimistic"])
    for role in REQUIRED_ROLES:
        assert guess[role].confidence == "exact"
    assert guess["optimistic"].column == "Optimistic"


def test_common_business_synonyms_map():
    guess = guess_mapping(
        ["Work package", "Best case", "Most likely", "Worst case"]
    )
    assert guess["task"].column == "Work package"
    assert guess["optimistic"].column == "Best case"
    assert guess["realistic"].column == "Most likely"
    assert guess["pessimistic"].column == "Worst case"


def test_russian_headers_map():
    guess = guess_mapping(
        ["Задача", "Оптимистичная", "Реалистичная", "Пессимистичная"]
    )
    assert guess["task"].column == "Задача"
    assert guess["realistic"].column == "Реалистичная"


def test_units_in_parentheses_are_ignored():
    guess = guess_mapping(["Activity", "Optimistic (days)", "Realistic (days)",
                           "Pessimistic (days)"])
    assert guess["optimistic"].column == "Optimistic (days)"


def test_a_typo_is_recovered_by_fuzzy_matching():
    guess = guess_mapping(["Task", "Optimisitc", "Realistic", "Pessimistic"])
    assert guess["optimistic"].column == "Optimisitc"
    assert guess["optimistic"].confidence == "fuzzy"


def test_unrecognisable_headers_produce_no_guess():
    guess = guess_mapping(["alpha", "beta", "gamma", "delta"])
    assert all(guess[role].column is None for role in ROLES)
    assert all(guess[role].confidence == "none" for role in ROLES)


def test_a_column_is_never_claimed_by_two_roles():
    guess = guess_mapping(["Estimate", "Estimate 2"])
    claimed = [g.column for g in guess.values() if g.column is not None]
    assert len(claimed) == len(set(claimed))


def test_every_role_is_present_in_the_result():
    assert set(guess_mapping(["anything"])) == set(ROLES)


def test_stream_is_not_a_recognised_role():
    """The Stream field was removed from the UI; it must not resurface."""
    assert "stream" not in ROLES


def test_short_realistic_abbreviation_maps_exactly():
    guess = guess_mapping(["Task", "min", "real", "max"])
    assert guess["realistic"].column == "real"
    assert guess["realistic"].confidence == "exact"


def test_header_row_is_detected_below_a_merged_section_title():
    """A merged header like 'Back-End' spanning min/real/max should not
    fool the detector into treating it as the real header row."""
    rows = [
        ["Обсяг робіт", "Back-End", None, None],
        ["Задача", "min", "real", "max"],
        ["Створення репозиторію проекту", 1, 1, 2],
    ]
    assert detect_header_row(rows) == 1


def test_header_row_is_zero_for_an_already_clean_sheet():
    rows = [
        ["Task", "Optimistic", "Realistic", "Pessimistic"],
        ["Discovery", 8, 12, 20],
    ]
    assert detect_header_row(rows) == 0


def test_header_row_falls_back_to_zero_when_nothing_matches():
    rows = [["a", "b", "c"], ["d", "e", "f"]]
    assert detect_header_row(rows) == 0


def test_header_row_ignores_blank_and_nan_cells():
    rows = [[None, float("nan"), ""], ["Task", "Optimistic", "Realistic"]]
    assert detect_header_row(rows) == 1
