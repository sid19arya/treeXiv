from treexiv.corpus import BM25Corpus, document_text, tokenize
from treexiv.models import Node


def _node(id_: str, title: str, abstract: str = "") -> Node:
    return Node(id=id_, title=title, publication_year=2020, cited_by_count=1,
                authors=[], venue=None, abstract=abstract, hop=0)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Recursive Language Models!") == ["recursive", "language", "models"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_document_text_combines_title_and_abstract() -> None:
    node = _node("W1", "Title Here", "abstract text")
    assert document_text(node) == "Title Here abstract text"


def test_top_k_ranks_relevant_document_first() -> None:
    # Three documents (not two) so the query terms' document frequency isn't
    # exactly half the corpus - at exactly N/2, BM25's classic idf term is 0
    # and every score collapses to 0 regardless of term frequency.
    nodes = [
        _node("W1", "Recursive language models for reasoning", "self-referential inference chains"),
        _node("W2", "Unrelated topic about gardening", "tomatoes and soil pH"),
        _node("W3", "A history of bridge engineering", "trusses and suspension cables"),
    ]
    corpus = BM25Corpus(nodes)
    ranked = corpus.top_k("recursive language models", k=3)
    assert ranked[0][0].id == "W1"
    assert ranked[0][1] > ranked[1][1]


def test_top_k_respects_k() -> None:
    nodes = [_node(f"W{i}", f"Paper {i} about recursion") for i in range(10)]
    corpus = BM25Corpus(nodes)
    assert len(corpus.top_k("recursion", k=3)) == 3


def test_scores_covers_every_node() -> None:
    nodes = [_node("W1", "A"), _node("W2", "B")]
    corpus = BM25Corpus(nodes)
    scores = corpus.scores("A")
    assert set(scores.keys()) == {"W1", "W2"}


def test_empty_corpus_returns_empty_scores() -> None:
    corpus = BM25Corpus([])
    assert corpus.scores("anything") == {}
    assert corpus.top_k("anything", k=5) == []
