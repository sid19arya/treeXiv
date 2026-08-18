from treexiv.abstract import reconstruct_abstract


def test_reconstructs_words_in_order() -> None:
    inverted = {"Hello": [0], "world": [1], "again": [2]}
    assert reconstruct_abstract(inverted) == "Hello world again"


def test_handles_repeated_words() -> None:
    inverted = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
    assert reconstruct_abstract(inverted) == "the cat sat the mat"


def test_none_returns_empty_string() -> None:
    assert reconstruct_abstract(None) == ""


def test_empty_dict_returns_empty_string() -> None:
    assert reconstruct_abstract({}) == ""


def test_non_contiguous_positions_leave_gaps_blank() -> None:
    inverted = {"start": [0], "end": [4]}
    assert reconstruct_abstract(inverted) == "start    end"
