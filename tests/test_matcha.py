import math
from collections import defaultdict
from typing import Callable, Set, List

import pytest
import spacy
from spacy.tokens import Span

from matcha import (
    ngrams,
    get_pos_fingerprint,
    get_lemmatized_phrase,
    is_phrase_matching,
    extract_terms,
    subphrases,
    calculate_nesting,
    combo_basic,
    basic,
    cvalue,
    upos_to_penn,
    extract_context,
)


class TestNgrams:
    """Test suite for Matcha NLP utilities."""

    @pytest.fixture
    def nlp(self) -> Callable:
        """Fixture providing spaCy model."""
        return spacy.load("en_core_web_sm", disable=["parser", "entity"])

    @pytest.fixture
    def basic_doc(self, nlp: Callable) -> spacy.tokens.Doc:
        """Fixture providing a basic test document."""
        return nlp("The big brown fox")

    def test_unigrams(self, basic_doc: spacy.tokens.Doc):
        """Test extraction of unigrams."""
        expected = ["The", "big", "brown", "fox"]
        spans = list(ngrams(basic_doc, n_min=1, n_max=1))

        assert len(spans) == len(expected)
        assert [span.text for span in spans] == expected

    def test_bigrams(self, basic_doc: spacy.tokens.Doc):
        """Test extraction of bigrams."""
        expected = ["The big", "big brown", "brown fox"]
        spans = list(ngrams(basic_doc, n_min=2, n_max=2))

        assert len(spans) == len(expected)
        assert [span.text for span in spans] == expected

    def test_trigrams(self, basic_doc: spacy.tokens.Doc):
        """Test extraction of trigrams."""
        expected = ["The big brown", "big brown fox"]
        spans = list(ngrams(basic_doc, n_min=3, n_max=3))

        assert len(spans) == len(expected)
        assert [span.text for span in spans] == expected

    def test_variable_length(self, basic_doc: spacy.tokens.Doc):
        """Test extraction of variable length n-grams."""
        spans = list(ngrams(basic_doc, n_min=1, n_max=4))
        span_counts = [len([s for s in spans if len(s) == i]) for i in range(1, 5)]
        expected_counts = [4, 3, 2, 1]
        assert span_counts == expected_counts

    def test_full_length(self, basic_doc: spacy.tokens.Doc):
        """Test extraction of n-grams with no n_max."""
        spans = list(ngrams(basic_doc, n_min=3))
        span_counts = [len([s for s in spans if len(s) == i]) for i in range(1, 5)]
        expected_counts = [0, 0, 2, 1]
        assert span_counts == expected_counts

    def test_empty_doc(self, nlp: Callable):
        """Test behavior with empty document."""
        doc = nlp("")
        spans = list(ngrams(doc, n_min=1, n_max=3))
        assert len(spans) == 0

    def test_invalid_n_min(self, basic_doc: spacy.tokens.Doc):
        """Test error handling for invalid n_min."""
        with pytest.raises(ValueError, match="n_min must be at least 1"):
            list(ngrams(basic_doc, n_min=0, n_max=2))

    def test_invalid_n_max(self, basic_doc: spacy.tokens.Doc):
        """Test error handling for invalid n_max."""
        with pytest.raises(
            ValueError, match="n_max must be greater than or equal to n_min"
        ):
            list(ngrams(basic_doc, n_min=2, n_max=1))


class TestFingerprintAndLemmatization:
    """Test suite for Matcha NLP utilities."""

    @pytest.fixture
    def nlp(self) -> Callable:
        """Fixture providing spaCy model."""
        return spacy.load("en_core_web_sm", disable=["parser", "entity"])

    @pytest.fixture
    def complex_doc(self, nlp: Callable) -> spacy.tokens.Doc:
        """Fixture providing a document with various POS and lemma cases."""
        return nlp("The running dogs are chasing cats quickly")

    def test_pos_fingerprint_basic(self, nlp: Callable):
        """Test basic POS fingerprint generation using Penn Treebank tags."""
        doc = nlp("The cat")
        span = doc[0:2]
        # DT for determiner, NN for singular noun
        assert get_pos_fingerprint(span) == "DT_NN_"

    def test_pos_fingerprint_complex(self, complex_doc: spacy.tokens.Doc):
        """Test POS fingerprint with various Penn Treebank tags."""
        # Get first four tokens: "The running dogs are"
        span = complex_doc[0:4]
        # DT for determiner, VBG for gerund, NNS for plural noun, VBP for non-3rd person singular present verb
        assert get_pos_fingerprint(span) == "DT_VBG_NNS_VBP_"

    def test_pos_fingerprint_single_token(self, complex_doc: spacy.tokens.Doc):
        """Test POS fingerprint for single token."""
        # Get just "quickly" - RB for adverb
        span = complex_doc[-1:]
        assert get_pos_fingerprint(span) == "RB_"

    def test_lemmatize_basic(self, nlp: Callable):
        """Test basic lemmatization with lowercasing."""
        doc = nlp("The Cats")
        span = doc[0:2]
        assert " ".join(get_lemmatized_phrase(span)) == "the cat"

    def test_lemmatize_complex(self, complex_doc: spacy.tokens.Doc):
        """Test lemmatization with various forms and lowercasing."""
        # "running dogs are chasing" -> "run dog be chase"
        span = complex_doc[1:5]
        assert " ".join(get_lemmatized_phrase(span)) == "run dog be chase"

    def test_lemmatize_single_token(self, complex_doc: spacy.tokens.Doc):
        """Test lemmatization for single token with lowercasing."""
        # "Running" -> "run"
        span = complex_doc[1:2]
        assert " ".join(get_lemmatized_phrase(span)) == "run"

    @pytest.mark.parametrize(
        "text,expected_fingerprint",
        [
            ("The cat", "DT_NN_"),
            ("Big red house", "JJ_JJ_NN_"),
            ("Quickly run", "RB_VB_"),
            ("The", "DT_"),
        ],
    )
    def test_pos_fingerprint_parametrized(
        self, nlp: Callable, text: str, expected_fingerprint: str
    ):
        """Test POS fingerprint with various Penn Treebank patterns."""
        doc = nlp(text)
        span = doc[0 : len(doc)]
        assert get_pos_fingerprint(span) == expected_fingerprint

    @pytest.mark.parametrize(
        "text,expected_lemma",
        [
            ("The cats are Running", "the cat be run"),
            ("Better Cities", "well city"),
            # Why on earth Fastest Cars are giving different results than fastest Cars?
            ("fastest cars", "fast car"),
            ("Is Going", "be go"),
        ],
    )
    def test_lemmatize_parametrized(
        self, nlp: Callable, text: str, expected_lemma: str
    ):
        """Test lemmatization with various input patterns and case variations."""
        doc = nlp(text)
        span = doc[0 : len(doc)]
        assert " ".join(get_lemmatized_phrase(span)) == expected_lemma


class TestIsPhraseMatching:
    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

    @pytest.mark.parametrize(
        "text,should_match",
        [
            # Valid patterns ending in noun
            ("dog", True),  # Simple noun
            ("big dog", True),  # Adjective + noun
            ("very big dog", False),  # Adverb breaks pattern
            # This is a valid pattern, but we do not support PP as ADJ yet
            # ("running dog", True),  # Present participle as adjective + noun
            ("artificial intelligence", True),  # Adjective + noun
            ("neural network", True),  # Adjective + noun
            ("deep neural network", True),  # Multiple adjectives + noun
            ("theory of computation", True),  # Noun + IN + noun
            ("database of knowledge", True),  # Noun + IN + noun
            ("statistical machine learning", True),  # Multiple adjectives + noun
            # Invalid patterns
            ("the dog", False),  # Determiner breaks pattern
            ("run quickly", False),  # Doesn't end in noun
            ("is good", False),  # Doesn't end in noun
            ("very nice", False),  # Doesn't end in noun
            ("the big red", False),  # Doesn't end in noun
            ("dogs and cats", False),  # Conjunction breaks pattern
            ("quickly running", False),  # Doesn't end in noun
        ],
    )
    def test_is_phrase_matching(self, nlp: Callable, text: str, should_match: bool):
        """Test phrase matching with various patterns."""
        doc = nlp(text)
        span = doc[0 : len(doc)]
        assert is_phrase_matching(span) == should_match

    def test_is_phrase_matching_edge_cases(self, nlp: Callable):
        """Test phrase matching with edge cases."""
        # Single proper noun
        doc = nlp("John")
        assert not is_phrase_matching(doc[0:1]), "Proper noun alone shouldn't match"

        # Single common noun
        doc = nlp("dog")
        assert is_phrase_matching(doc[0:1]), "Common noun alone should match"

        # Disabled the detection of single word adjs for now.
        ## Single adj should match if single word is allowed
        doc = nlp("Playful")
        # assert is_phrase_matching(
        #     doc[0:1], allow_single_word=True
        # ), "Adj alone should match if allowed"
        assert not is_phrase_matching(doc[0:1]), "Adj alone shouldn't match"

        # Complex noun phrase with preposition
        doc = nlp("analysis of neural networks")
        assert is_phrase_matching(doc[0:4]), "Noun + IN + adjective + noun should match"

        # ... but not if started with hyphen
        doc = nlp("- analysis of neural networks")
        assert not is_phrase_matching(doc[0:5]), "HYPH is not allowed at the beginning!"

        doc = nlp("front-end and back-end")
        assert is_phrase_matching(doc[0:3]), "Compound noun with hyphen should match"
        assert is_phrase_matching(doc[4:7]), "Compound noun with hyphen should match"

        doc = nlp("Short-term")
        assert is_phrase_matching(
            doc[0:3]
        ), "Compound adjectives with hyphen should match too"

        # Multiple adjectives
        doc = nlp("deep convolutional neural network")
        assert is_phrase_matching(
            doc[0:4]
        ), "Multiple adjectives ending in noun should match"

    def test_is_phrase_matching_nested_spans(self, nlp: Callable):
        """Test phrase matching with nested term candidates."""
        doc = nlp("artificial neural network architecture")

        # Test various nested spans
        assert is_phrase_matching(doc[0:4]), "Full span should match"
        assert is_phrase_matching(
            doc[0:3]
        ), "Nested 'artificial neural network' should match"
        assert is_phrase_matching(doc[1:3]), "Nested 'neural network' should match"
        assert is_phrase_matching(
            doc[2:4]
        ), "Nested 'network architecture' should match"


class TestExtractTerms:
    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def spans_to_texts(self, spans):
        return [span.text for span in spans]

    def test_extract_terms_basic(self, nlp: Callable):
        """Test basic term extraction with fox sentence."""
        doc = nlp("The big brown fox jumps over the lazy dog")
        freq, occurrences = extract_terms(doc)

        # Only valid terms from the sentence should be:
        # - "big brown fox" (JJ JJ NN)
        # - "brown fox" (JJ NN)
        # - "lazy dog" (JJ NN)

        assert len(freq) == 3, "Should find exactly 3 terms"
        assert freq["big brown fox"] == 1
        assert freq["brown fox"] == 1
        assert freq["lazy dog"] == 1

        assert self.spans_to_texts(occurrences["big brown fox"]) == ["big brown fox"]
        assert self.spans_to_texts(occurrences["brown fox"]) == ["brown fox"]
        assert self.spans_to_texts(occurrences["lazy dog"]) == ["lazy dog"]

    def test_extract_terms_neural(self, nlp: Callable):
        """Test term extraction with neural network example."""
        doc = nlp("artificial neural network architecture")
        freq, occurrences = extract_terms(doc)

        # Expected valid terms:
        # - "artificial neural network architecture"
        # - "artificial neural network"
        # - "neural network architecture"
        # - "neural network"
        # - "network architecture"

        assert len(freq) == 5, "Should find exactly 5 terms"

        expected_terms = {
            "artificial neural network architecture",
            "artificial neural network",
            "neural network architecture",
            "neural network",
            "network architecture",
        }

        assert set(freq.keys()) == expected_terms
        assert all(freq[term] == 1 for term in freq), "All terms should occur once"
        assert all(
            len(occurrences[term]) == 1 for term in occurrences
        ), "All terms should have one surface form"

    def test_extract_terms_repeated(self, nlp: Callable):
        """Test term extraction with repeated terms."""
        doc = nlp("The neural network processes data using a neural network model")
        freq, occurrences = extract_terms(doc)

        assert (
            freq["neural network"] == 2
        ), "Should count two occurrences of 'neural network'"
        assert self.spans_to_texts(occurrences["neural network"]) == [
            "neural network",
            "neural network",
        ]
        assert freq["neural network model"] == 1

    def test_extract_terms_with_signle_words(self, nlp: Callable):
        """Test term extraction with repeated terms."""
        doc = nlp("The neural network processes data using a neural network model")
        freq, _ = extract_terms(doc, n_min=1)

        # Disabled the detection of single word adjs for now.
        # assert freq["neural"] == 2, "Should count two occurrences of 'neural'"
        assert freq["network"] == 2, "Should count two occurrences of 'network'"
        assert freq["datum"] == 1, "Should count 1 occurrences of lemmatized 'data'"
        assert freq["model"] == 1, "Should count two occurrences of 'model'"
        assert freq["processes"] == 0, "Should find none of verbs"

        assert (
            freq["neural network"] == 2
        ), "Should count two occurrences of 'neural network'"
        assert freq["neural network model"] == 1

    def test_extract_terms_case_variations(self, nlp: Callable):
        """Test term extraction with case variations."""
        doc = nlp("The Neural network uses another neural network")
        freq, occurrences = extract_terms(doc)

        # Should normalize to same lemma despite case difference
        assert freq["neural network"] == 2
        assert self.spans_to_texts(occurrences["neural network"]) == [
            "Neural network",
            "neural network",
        ]

    def test_extract_terms_length_limits(self, nlp: Callable):
        """Test term extraction with different length limits."""
        doc = nlp("deep artificial neural network architecture")

        # Test with max_length=2
        freq, _ = extract_terms(doc, n_max=2)
        assert "neural network" in freq
        assert "deep artificial" not in freq
        assert "artificial neural network" not in freq

        # Test with min_length=3
        freq, _ = extract_terms(doc, n_min=3)
        assert "neural network" not in freq
        assert "deep artificial neural" not in freq
        assert "artificial neural network" in freq
        assert "deep artificial neural network" in freq
        assert "deep artificial neural network architecture" in freq

    def test_extract_terms_empty_doc(self, nlp: Callable):
        """Test term extraction with empty document."""
        doc = nlp("")
        freq, occurrences = extract_terms(doc)

        assert len(freq) == 0
        assert len(occurrences) == 0

    def test_extract_terms_no_valid_terms(self, nlp: Callable):
        """Test term extraction with text containing no valid terms."""
        doc = nlp("The and or if but")  # Only stopwords/function words
        freq, occurrences = extract_terms(doc)

        assert len(freq) == 0
        assert len(occurrences) == 0

    @pytest.mark.parametrize(
        "text,expected_terms",
        [
            # Technical terms
            (
                "deep learning model architecture",
                {
                    "deep learning model architecture",
                    "deep learning model",
                    "learning model architecture",
                    "deep learning",
                    "learning model",
                    "model architecture",
                },
            ),
            # Noun phrases with prepositions
            ("theory of computation", {"theory of computation"}),
            # Multiple adjectives
            (
                "statistical machine learning",
                {
                    "statistical machine learning",
                    "statistical machine",
                    "machine learning",
                },
            ),
        ],
    )
    def test_extract_terms_parametrized(
        self, nlp: Callable, text: str, expected_terms: Set[str]
    ):
        """Test term extraction with various input patterns."""
        doc = nlp(text)
        freq, _ = extract_terms(doc)
        assert set(freq.keys()) == expected_terms


class TestStopwords:
    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def test_extract_terms_with_stopwords(self, nlp: Callable):
        """Test term extraction with stopwords filtering."""
        doc = nlp("The neural network of machine learning")
        stopwords = {"of"}

        # Without stopwords filtering
        freq1, _ = extract_terms(doc)
        assert "neural network" in freq1
        assert "machine learning" in freq1
        assert "neural network of machine learning" in freq1

        # With stopwords filtering
        freq2, _ = extract_terms(doc, stopwords=stopwords)
        assert "neural network" in freq2  # Should keep
        assert "machine learning" in freq2  # Should keep
        assert "neural network of machine learning" not in freq2  # Should filter out

    def test_extract_terms_stopwords_case_insensitive(self, nlp: Callable):
        """Test stopwords filtering is case insensitive."""
        doc = nlp("The neural network Of deep learning")
        stopwords = {"the", "of"}  # lowercase stopwords

        freq, _ = extract_terms(doc, stopwords=stopwords)
        assert "neural network" in freq  # Should keep
        assert "deep learning" in freq  # Should keep
        assert "neural network of deep learning" not in freq  # Should filter out

    def test_extract_terms_empty_stopwords(self, nlp: Callable):
        """Test term extraction with empty stopwords set."""
        doc = nlp("The neural network of deep learning")

        # Empty set of stopwords should be same as no stopwords
        freq1, occ1 = extract_terms(doc, stopwords=set())
        freq2, occ2 = extract_terms(doc)

        assert freq1 == freq2
        assert occ1 == occ2

    @pytest.mark.parametrize(
        "text,stopwords,expected_terms",
        [
            # Basic stopword filtering
            (
                "artificial neural network of science",
                {"of"},
                {"artificial neural network", "neural network"},
            ),
            # Multiple stopwords
            ("theory of quantum computation", {"of", "the"}, {"quantum computation"}),
            # No matching stopwords
            (
                "deep machine learning",
                {"of", "the"},
                {
                    "deep machine learning",
                    "machine learning",
                    "deep machine",
                },
            ),
        ],
    )
    def test_extract_terms_stopwords_parametrized(
        self, nlp: Callable, text: str, stopwords: Set[str], expected_terms: Set[str]
    ):
        """Test stopwords filtering with various patterns."""
        doc = nlp(text)
        freq, _ = extract_terms(doc, stopwords=stopwords)
        assert set(freq.keys()) == expected_terms


class TestSubphrases:
    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def test_string_input(self):
        """Test with basic string input."""
        result = list(subphrases("artificial neural network"))
        expected = [
            "artificial neural",
            "neural network",
        ]
        assert result == expected

    def test_doc_input(self, nlp):
        """Test with spaCy Doc input."""
        doc = nlp("artificial neural network")
        result = list(subphrases(doc))
        expected = [
            "artificial neural",
            "neural network",
        ]
        assert result == expected

    def test_span_input(self, nlp):
        """Test with spaCy Span input."""
        doc = nlp("the artificial neural network architecture")
        span = doc[1:4]  # "artificial neural network"
        result = list(subphrases(span))
        expected = [
            "artificial neural",
            "neural network",
        ]
        assert result == expected

    def test_different_min_length(self):
        """Test with different minimum length settings."""
        phrase = "deep neural network architecture"
        result_min2 = list(subphrases(phrase, n_min=2))  # default
        result_min3 = list(subphrases(phrase, n_min=3))

        # Check min=2 includes both bigrams and trigrams
        assert {
            "deep neural",
            "neural network",
            "network architecture",
            "deep neural network",
            "neural network architecture",
        } == set(result_min2)

        # Check min=3 only includes trigrams and up
        assert {
            "deep neural network",
            "neural network architecture",
        } == set(result_min3)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty string
        assert list(subphrases("")) == []

        # Single word
        assert list(subphrases("word")) == []

        # Two words
        assert list(subphrases("two words")) == []

    def test_no_full_sequence(self):
        """Test that full sequence is not included in results."""
        phrase = "deep neural network architecture"
        result = list(subphrases(phrase))
        assert "deep neural network" in result

        # Test no longer sequences
        assert all(len(subseq.split()) < 4 for subseq in result)

    def test_input_equivalence(self, nlp):
        """Test that different input types produce same results."""
        text = "artificial neural network"
        doc = nlp(text)
        span = doc[0:3]

        str_result = list(subphrases(text))
        doc_result = list(subphrases(doc))
        span_result = list(subphrases(span))

        assert str_result == doc_result == span_result

    @pytest.mark.parametrize(
        "phrase,n_min,expected",
        [
            # Basic bigrams and trigram
            (
                "neural network model",
                2,
                ["neural network", "network model"],
            ),
            # Empty results
            ("neural network architecture", 3, []),
            ("single", 2, []),
            ("single word", 2, []),
        ],
    )
    def test_parametrized(self, phrase: str, n_min: int, expected: List[List[str]]):
        """Test various input combinations."""
        result = list(subphrases(phrase, n_min=n_min))
        assert result == expected


class TestNestingMetrics:
    """Test suite for term nesting metrics calculation."""

    def test_basic_nesting(self):
        """Test basic nesting counting without frequencies."""
        terms = {"neural network": 1, "deep neural network": 1}

        supersets, subsets = calculate_nesting(terms)

        # neural network appears in deep neural network once
        assert supersets["neural network"] == 1
        # deep neural network contains neural network once
        assert subsets["deep neural network"] == 1
        # neural network contains no terms (as we only consider multi-word phrases)
        assert subsets["neural network"] == 0
        # deep neural network appears in no terms
        assert supersets["deep neural network"] == 0

    def test_multiple_nesting(self):
        """Test when a term appears in multiple other terms."""
        terms = {
            "neural network": 1,
            "deep neural network": 1,
            "artificial neural network": 1,
            "deep neural network architecture": 1,
        }

        supersets, subsets = calculate_nesting(terms)

        # 'neural network' appears in 3 longer terms
        assert supersets["neural network"] == 3
        # 'deep neural network' appears in 1 term
        assert supersets["deep neural network"] == 1

        # The longer terms contain multi-word subterms
        assert subsets["deep neural network"] == 1  # contains 'neural network'
        assert subsets["artificial neural network"] == 1  # contains 'neural network'
        assert (
            subsets["deep neural network architecture"] == 2
        )  # contains 'neural network' and 'deep neural network'

    def test_frequency_weighting(self):
        """Test nesting counting with frequency weighting."""
        terms = {
            "neural network": 5,  # appears five times
            "deep neural network": 3,  # appears three times
        }

        supersets, subsets = calculate_nesting(terms, use_frequencies=True)

        # neural network (freq=5) appears in deep neural network (freq=3)
        # weight = 3
        assert supersets["neural network"] == 3

        # deep neural network contains neural network
        # weight = 3
        assert subsets["deep neural network"] == 3

    def test_no_nesting(self):
        """Test with terms that have no nesting relationships."""
        terms = {"neural network": 1, "deep learning": 1, "artificial intelligence": 1}

        supersets, subsets = calculate_nesting(terms)

        # No term should have any nesting relationships
        assert sum(supersets.values()) == 0
        assert sum(subsets.values()) == 0

    def test_multiple_levels(self):
        """Test nested terms of different lengths."""
        terms = {
            "neural network": 1,
            "deep neural network": 1,
            "artificial neural network": 1,
            "very deep neural network": 1,
        }

        supersets, subsets = calculate_nesting(terms)

        # Check number of appearances
        assert supersets["neural network"] == 3  # appears in all longer terms
        assert (
            supersets["deep neural network"] == 1
        )  # appears in 'very deep neural network'
        assert supersets["artificial neural network"] == 0  # appears in no terms
        assert supersets["very deep neural network"] == 0  # appears in no terms

        # Check number of contained terms
        assert subsets["neural network"] == 0  # contains no multi-word terms
        assert subsets["deep neural network"] == 1  # contains 'neural network'
        assert subsets["artificial neural network"] == 1  # contains 'neural network'
        assert (
            subsets["very deep neural network"] == 2
        )  # contains 'neural network' and 'deep neural network'

    def test_empty_input(self):
        """Test with empty input."""
        supersets, subsets = calculate_nesting({})

        assert isinstance(supersets, defaultdict)
        assert isinstance(subsets, defaultdict)
        assert len(supersets) == 0
        assert len(subsets) == 0

    def test_single_term(self):
        """Test with single term."""
        terms = {"neural network": 1}

        supersets, subsets = calculate_nesting(terms)

        assert supersets["neural network"] == 0
        assert subsets["neural network"] == 0


class TestComboBasic:
    """Test suite for ComboBasic term scoring."""

    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "entity"])

    def test_basic_scoring(self, nlp):
        """Test basic scoring functionality."""

        # Adding ellipsis here to not match neural network neural network, which is
        # a legit term according to our pos_fingerprint pattern
        doc = nlp("neural network... neural network... deep neural network")
        scores, _ = combo_basic(doc)

        # neural network: length=2, freq=2, appears in 1 term
        # score = 2 * log(2) + 0.75 * 1 + 0.1 * 0
        expected_nn = 2 * math.log(2) + 0.75 * 1
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)

        # deep neural network: length=3, freq=1, contains 1 term
        # score = 3 * log(1) + 0.75 * 0 + 0.1 * 1
        expected_dnn = 0 + 0.1 * 1  # log(1) = 0
        assert math.isclose(scores["deep neural network"], expected_dnn, rel_tol=1e-10)

    def test_stopwords(self, nlp):
        """Test scoring with stopwords filtering."""
        doc = nlp("the neural network and deep neural network or neural of the network")
        stopwords = {"the", "and", "of"}
        scores, _ = combo_basic(doc, stopwords=stopwords)

        assert "neural network" in scores
        assert "deep neural network" in scores
        # Should be same scores as without stopwords
        expected_nn = 2 * math.log(2) + 0.75 * 1
        expected_dnn = 3 * math.log(1) + 0.1 * 1
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)
        assert math.isclose(scores["deep neural network"], expected_dnn, rel_tol=1e-10)

    def test_custom_parameters(self, nlp):
        """Test scoring with custom alpha and beta."""
        doc = nlp("neural network... neural network... deep neural network")
        scores, _ = combo_basic(doc, alpha=1.0, beta=0.5)

        # neural network: length=2, freq=2, appears in 1 term
        # score = 2 * log(2) + 1.0 * 1 + 0.5 * 0
        expected_nn = 2 * math.log(2) + 1.0 * 1
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)

        # deep neural network: length=3, freq=1, contains 1 term
        # score = 3 * log(1) + 1.0 * 0 + 0.5 * 1
        expected_dnn = 0 + 0.5 * 1  # log(1) = 0
        assert math.isclose(scores["deep neural network"], expected_dnn, rel_tol=1e-10)

    def test_length_limits(self, nlp):
        """Test scoring with different length limits."""
        doc = nlp("neural network deep neural network very deep neural network")

        # Test with n_max=2
        scores_max2, _ = combo_basic(doc, n_max=2)
        assert "neural network" in scores_max2
        assert "deep neural network" not in scores_max2

        # Test with n_min=3
        scores_min3, _ = combo_basic(doc, n_min=3)
        assert "neural network" not in scores_min3
        assert "deep neural network" in scores_min3

    def test_frequency_weighting(self, nlp):
        """Test scoring with frequency weighting in nesting calculations."""
        doc = nlp(
            "neural network... neural network... deep neural network... deep neural network"
        )
        scores, _ = combo_basic(doc, use_frequencies=True)

        # neural network appears twice in deep neural network which appears twice
        # With frequency weighting, supersets count should be 2 instead of 1
        expected_nn = 2 * math.log(3) + 0.75 * 2
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)

    def test_smoothing(self, nlp):
        """Test scoring with frequency smoothing in nesting calculations."""
        doc = nlp(
            "neural network... neural network... deep neural network... deep neural network"
        )
        scores, _ = combo_basic(doc, smoothing=2)

        expected_nn = 2 * math.log(3 + 2) + 0.75 * 1
        expected_dnn = 3 * math.log(2 + 2) + 0.1 * 1

        # Here is dnn score is higher because smoothing is applied
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)
        assert math.isclose(scores["deep neural network"], expected_dnn, rel_tol=1e-10)

    def test_empty_doc(self, nlp):
        """Test scoring with empty document."""
        doc = nlp("")
        scores, _ = combo_basic(doc)
        assert len(scores) == 0


class TestBasic:
    """Test suite for Basic term scoring."""

    @pytest.fixture
    def nlp(self):
        return spacy.load("en_core_web_sm", disable=["parser", "entity"])

    def test_basic_scoring(self, nlp):
        """Test basic scoring functionality."""

        # Adding ellipsis here to not match neural network neural network, which is
        # a legit term according to our pos_fingerprint pattern
        doc = nlp("neural network... neural network... deep neural network")
        scores, _ = basic(doc)

        # neural network: length=2, freq=2, appears in 1 term
        # score = 2 * log(2) + 0.72 * 1
        expected_nn = 2 * math.log(2) + 0.72 * 1
        assert math.isclose(scores["neural network"], expected_nn, rel_tol=1e-10)

        # deep neural network: length=3, freq=1, contains 1 term
        # score = 3 * log(1) + 0.72 * 0
        expected_dnn = 0
        assert math.isclose(scores["deep neural network"], expected_dnn, rel_tol=1e-10)


class TestCValue:
    """Test suite for C-Value term scoring."""

    @pytest.fixture
    def nlp(self):
        """Provide spaCy model."""
        return spacy.load("en_core_web_sm", disable=["parser", "entity"])

    def test_non_nested_terms(self, nlp):
        """Test C-Value calculation for non-nested terms."""
        doc = nlp("neural network... deep learning")
        scores, _ = cvalue(doc)

        # For "neural network":
        # - length = 2, freq = 1, not nested
        # score = log2(2 + 0.1) * 1 = log2(2.1) * 1
        expected_score = math.log2(2.1)
        assert math.isclose(scores["neural network"], expected_score, rel_tol=1e-10)

        # Same for "deep learning"
        assert math.isclose(scores["deep learning"], expected_score, rel_tol=1e-10)

    def test_nested_terms(self, nlp):
        """Test C-Value calculation for nested terms."""
        doc = nlp("neural network, the neural network and the deep neural network")
        scores, _ = cvalue(doc)

        # For "neural network":
        # - length = 2, freq = 2
        # - appears in "deep neural network" (freq = 1)
        # score = log2(2.1) * (2 - 1/1)
        expected_score = math.log2(2.1) * (2 - 1)
        assert math.isclose(scores["neural network"], expected_score, rel_tol=1e-10)

        # For "deep neural network":
        # - length = 3, freq = 1, not nested
        # score = log2(3.1) * 1
        expected_score_dnn = math.log2(3.1)
        assert math.isclose(
            scores["deep neural network"], expected_score_dnn, rel_tol=1e-10
        )

    def test_multiple_nested_occurrences(self, nlp):
        """Test C-Value with multiple containing terms."""
        doc = nlp(
            "a neural network, the deep neural network and the artificial neural network"
        )
        scores, _ = cvalue(doc)

        # For "neural network":
        # - length = 2, freq = 3
        # - appears in two terms with freq=1 each
        # score = log2(2.1) * (3 - (1 + 1)/2)
        expected_score = math.log2(2.1) * (3 - 1.0)  # Should be 0
        assert math.isclose(scores["neural network"], expected_score, rel_tol=1e-10)

    def test_frequency_impact(self, nlp):
        """Test how term frequency affects the score."""
        doc = nlp("the neural network, the neural network, the neural network")
        scores, _ = cvalue(doc)

        # For "neural network":
        # - length = 2, freq = 3, not nested
        # score = log2(2.1) * 3
        expected_score = math.log2(2.1) * 3
        assert math.isclose(scores["neural network"], expected_score, rel_tol=1e-10)

    def test_empty_doc(self, nlp):
        """Test C-Value with empty document."""
        doc = nlp("")
        scores, _ = cvalue(doc)
        assert len(scores) == 0

    def test_stopwords(self, nlp):
        """Test C-Value with stopwords filtering."""
        doc = nlp("the neural network and deep neural network")
        stopwords = {"the", "and"}
        scores, _ = cvalue(doc, stopwords=stopwords)

        assert "neural network" in scores
        assert "deep neural network" in scores
        assert len(scores) == 2  # Only these two terms should remain

    def test_length_limits(self, nlp):
        """Test C-Value with different length limits."""
        doc = nlp(
            "neural network deep neural network artificial neural network architecture"
        )

        # Test with n_max=2
        scores_max2, _ = cvalue(doc, n_max=2)
        assert "neural network" in scores_max2
        assert "deep neural network" not in scores_max2

        # Test with n_min=3
        scores_min3, _ = cvalue(doc, n_min=3)
        assert "neural network" not in scores_min3
        assert "deep neural network" in scores_min3


class TestUPOSToPenn:
    """Test cases for UPOS to Penn tag conversion."""

    def test_basic_conversion(self):
        """Test basic UPOS to Penn conversion."""
        input_tags = ["NOUN", "ADJ", "PROPN", "ADP"]
        expected = ["NN", "JJ", "NNP", "IN"]
        assert upos_to_penn(input_tags) == expected

    def test_unknown_tags(self):
        """Test UPOS to Penn conversion with unknown tags."""
        input_tags = ["NOUN", "UNKNOWN", "PROPN", "XX"]
        expected = ["NN", "UNKNOWN", "NNP", "XX"]
        assert upos_to_penn(input_tags) == expected

    def test_empty_list(self):
        """Test UPOS to Penn conversion with empty input."""
        assert upos_to_penn([]) == []


class TestContextWindow:
    """Test suite for context window extraction."""

    @pytest.fixture
    def nlp(self) -> Callable:
        """Fixture providing spaCy model."""
        return spacy.load("en_core_web_sm")  # , disable=["parser", "entity"])

    def test_extract_single_sentence(self, nlp):
        """Test extracting only the sentence where span belongs (n=1)."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span in sentence two
        span = doc[5:9]  # "is sentence two"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=1)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        assert len(surrounding_sents) == 1
        assert surrounding_sents[0].text == "This is sentence two."
        assert original_span == span

    def test_extract_two_sentences(self, nlp):
        """Test extracting span's sentence plus one before (n=2)."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span in sentence two
        span = doc[5:9]  # "is sentence two"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=2)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        assert len(surrounding_sents) == 2
        assert surrounding_sents[0].text == "This is sentence one."
        assert surrounding_sents[1].text == "This is sentence two."
        assert original_span == span

    def test_extract_three_sentences(self, nlp):
        """Test extracting span's sentence plus one before and after (n=3)."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span in sentence two
        span = doc[5:9]  # "is sentence two"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=3)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        assert len(surrounding_sents) == 3
        assert surrounding_sents[0].text == "This is sentence one."
        assert surrounding_sents[1].text == "This is sentence two."
        assert surrounding_sents[2].text == "This is sentence three."
        assert original_span == span

    def test_beginning_of_document(self, nlp):
        """Test behavior when span is at the beginning of the document (n=2)."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span in sentence one
        span = doc[0:4]  # "This is sentence one"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=2)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        # Should include first sentence and the next one
        assert len(surrounding_sents) == 2
        assert surrounding_sents[0].text == "This is sentence one."
        assert surrounding_sents[1].text == "This is sentence two."
        assert original_span == span

    def test_end_of_document(self, nlp):
        """Test behavior when span is at the end of the document (n=2)."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span in sentence three
        span = doc[10:14]  # "This is sentence three"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=2)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        # Should include last sentence and the previous one
        assert len(surrounding_sents) == 2
        assert surrounding_sents[0].text == "This is sentence two."
        assert surrounding_sents[1].text == "This is sentence three."
        assert original_span == span

    def test_cross_sentence_span(self, nlp):
        """Test behavior when span crosses sentence boundaries."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        doc = nlp(text)

        # Create a span that crosses from sentence one to sentence two
        span = Span(doc, 3, 7)  # "sentence one. This is"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=1)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        # Should include both sentences the span crosses
        assert len(surrounding_sents) == 2
        assert surrounding_sents[0].text == "This is sentence one."
        assert surrounding_sents[1].text == "This is sentence two."
        assert original_span == span

    def test_multiple_cross_sentence_span(self, nlp):
        """Test behavior when span crosses multiple sentence boundaries."""
        text = "This is one. This is two. This is three. This is four."
        doc = nlp(text)

        # Create a span that crosses from sentence one through three
        span = Span(doc, 2, 11)  # "one. This is two. This is"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=2)

        assert len(result["lemma1"]) == 1
        surrounding_sents, original_span = result["lemma1"][0]

        # Should include the three sentences the span crosses, plus one more to reach n=4
        assert len(surrounding_sents) >= 3
        assert surrounding_sents[0].text == "This is one."
        assert surrounding_sents[1].text == "This is two."
        assert surrounding_sents[2].text == "This is three."
        assert original_span == span

    def test_multiple_lemmas_and_spans(self, nlp):
        """Test with multiple lemmas and spans."""
        text = "This is one. This is two. This is three. This is four. This is five."
        doc = nlp(text)

        # Create multiple spans for different lemmas
        span1 = doc[2:3]  # "one"
        span2 = doc[6:7]  # "two"
        span3 = doc[14:15]  # "four"

        lemma_spans = {"lemma1": [span1, span3], "lemma2": [span2]}

        result = extract_context(doc, lemma_spans, n=1)

        # Check lemma1 results
        assert len(result["lemma1"]) == 2
        assert len(result["lemma1"][0][0]) == 1  # First span has 1 sentence
        assert result["lemma1"][0][0][0].text == "This is one."
        assert result["lemma1"][0][1] == span1

        assert len(result["lemma1"][1][0]) == 1  # Second span has 1 sentence
        assert result["lemma1"][1][0][0].text == "This is four."
        assert result["lemma1"][1][1] == span3

        # Check lemma2 results
        assert len(result["lemma2"]) == 1
        assert len(result["lemma2"][0][0]) == 1
        assert result["lemma2"][0][0][0].text == "This is two."
        assert result["lemma2"][0][1] == span2

    def test_empty_doc(self, nlp):
        """Test behavior with empty document."""
        doc = nlp("")
        lemma_spans = {"lemma1": []}

        result = extract_context(doc, lemma_spans, n=1)

        assert "lemma1" in result
        assert result["lemma1"] == []

    def test_multiple_cross_sentence_span_with_long_context(self, nlp):
        """Test behavior when span crosses multiple sentence boundaries."""
        text = (
            "This is zero. This is one. This is two. This is three. This is four. "
            + "This is five. This is six. This is seven. This is eight. This is nine."
        )
        doc = nlp(text)

        # Create a span that crosses from sentence one through three
        span = Span(doc, 14, 23)  # "three. This is four. This is five"
        lemma_spans = {"lemma1": [span]}

        result = extract_context(doc, lemma_spans, n=1)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 3
        assert surrounding_sents[0].text == "This is three."
        assert surrounding_sents[1].text == "This is four."
        assert surrounding_sents[2].text == "This is five."

        result = extract_context(doc, lemma_spans, n=2)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 4
        assert surrounding_sents[0].text == "This is two."
        assert surrounding_sents[1].text == "This is three."
        assert surrounding_sents[2].text == "This is four."
        assert surrounding_sents[3].text == "This is five."

        result = extract_context(doc, lemma_spans, n=3)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 5
        assert surrounding_sents[0].text == "This is two."
        assert surrounding_sents[1].text == "This is three."
        assert surrounding_sents[2].text == "This is four."
        assert surrounding_sents[3].text == "This is five."
        assert surrounding_sents[4].text == "This is six."

        result = extract_context(doc, lemma_spans, n=4)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 6
        assert surrounding_sents[0].text == "This is one."
        assert surrounding_sents[1].text == "This is two."
        assert surrounding_sents[2].text == "This is three."
        assert surrounding_sents[3].text == "This is four."
        assert surrounding_sents[4].text == "This is five."
        assert surrounding_sents[5].text == "This is six."

        result = extract_context(doc, lemma_spans, n=5)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 7
        assert surrounding_sents[0].text == "This is one."
        assert surrounding_sents[1].text == "This is two."
        assert surrounding_sents[2].text == "This is three."
        assert surrounding_sents[3].text == "This is four."
        assert surrounding_sents[4].text == "This is five."
        assert surrounding_sents[5].text == "This is six."
        assert surrounding_sents[6].text == "This is seven."

        result = extract_context(doc, lemma_spans, n=6)

        surrounding_sents, _ = result["lemma1"][0]

        assert len(surrounding_sents) == 8
        assert surrounding_sents[0].text == "This is zero."
        assert surrounding_sents[1].text == "This is one."
        assert surrounding_sents[2].text == "This is two."
        assert surrounding_sents[3].text == "This is three."
        assert surrounding_sents[4].text == "This is four."
        assert surrounding_sents[5].text == "This is five."
        assert surrounding_sents[6].text == "This is six."
        assert surrounding_sents[7].text == "This is seven."
