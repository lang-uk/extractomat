from typing import List, Union, Optional, Dict

import spacy
from spacy.tokens import Doc, Span
import numpy as np
from sentence_transformers import SentenceTransformer, util

from matcha import extract_context


class SentenceSimilarityCalculator:
    """
    A class that uses sentence-transformers (SBERT) to calculate
    cosine similarity between concatenated context sentences and a spaCy Span.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """
        Initialize the similarity calculator with a specific SBERT model.

        Args:
            model_name: Name of the sentence-transformers model to load
            device: Device to use for model inference ('cpu', 'cuda', etc.)
                    If None, will use cuda if available
        """
        self.model_name = model_name
        self.device = device

        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name, device=device)

    def get_embedding(self, text: Union[str, Span, Doc]) -> np.ndarray:
        """
        Get embedding for a text, Span, or Doc.

        Args:
            text: Input text to embed (string, Span, or Doc)

        Returns:
            Embedding as numpy array
        """
        if isinstance(text, (Span, Doc)):
            # Convert spaCy object to string
            text_str = text.text
        else:
            text_str = text

        # Get embedding from model
        return self.model.encode(text_str, show_progress_bar=False)

    def calculate_similarity(
        self,
        context_sentences: List[Union[str, Doc, Span]],
        target_span: Span,
        length_adjust: bool = True,
        length_factor: float = 0.2,
        min_tokens: int = 3,
        max_boost: float = 0.5,
    ) -> float:
        """
        Calculate cosine similarity between concatenated context sentences and the target span,
        with optional non-linear adjustment for shorter spans.

        Args:
            context_sentences: List of sentences to concatenate and compare against (as strings or spaCy objects)
            target_span: spaCy Span to compare with the concatenated context
            length_adjust: Whether to apply length adjustment to boost scores for shorter spans
            length_factor: Controls the strength of the length adjustment (higher = more adjustment)
            min_tokens: Minimum number of tokens that can receive adjustment
            max_boost: Maximum boost that can be applied to the similarity score

        Returns:
            Similarity score between the concatenated context and target span
        """
        # Concatenate all context sentences
        if all(isinstance(sent, str) for sent in context_sentences):
            # If all items are strings, join with spaces
            concatenated_context = " ".join(context_sentences)
        else:
            # If spaCy objects, extract text first
            concatenated_context = " ".join(
                [
                    sent.text if hasattr(sent, "text") else str(sent)
                    for sent in context_sentences
                ]
            )

        # Get embedding for the concatenated context
        context_embedding = self.get_embedding(concatenated_context)

        # Get embedding for the target span
        target_embedding = self.get_embedding(target_span)

        # Calculate cosine similarity using sentence_transformers util
        base_similarity = util.cos_sim(context_embedding, target_embedding).item()

        # Apply non-linear adjustment for short spans if requested
        if length_adjust and len(target_span) <= min_tokens:
            # Calculate a boosting factor based on span length
            # The shorter the span, the more boost (with diminishing returns)
            token_count = len(target_span)
            length_boost = max_boost * np.exp(-length_factor * token_count)

            # Apply the boost, ensuring we don't exceed 1.0
            adjusted_similarity = min(1.0, base_similarity + length_boost)
            return adjusted_similarity

        return base_similarity

    def rerank_terms_in_doc(
        self,
        doc: Doc,
        term_occurences=Dict[str, List[Span]],
        context_len: int = 3,
        pooling="max",
    ) -> Dict[str, float]:
        """
        Rerank terms in a document based on their similarity to the context.

        Args:
            doc: spaCy Doc object
            term_occurences: Dictionary of term occurences (spans) in the document
            context_len: Number of context sentences to consider
            pooling: Pooling strategy for term embeddings (max, mean, "
        Returns:
            Lemma-score dict as Dict[str, float]
        """
        assert pooling in ["max", "mean"], "Pooling must be 'max' or 'mean'"

        # Extract context sentences
        context_sentences = extract_context(doc, term_occurences, context_len)
        # print(context_sentences)

        # Calculate similarity for each term
        term_scores = {}
        for term, occurences in context_sentences.items():
            term_score = []
            for context, span in occurences:
                term_score.append(
                    self.calculate_similarity(
                        context_sentences=context, target_span=span
                    )
                )

            if pooling == "max":
                term_scores[term] = max(term_score)
            else:
                term_scores[term] = sum(term_score) / len(occurences)

        return term_scores


# Example usage
if __name__ == "__main__":
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Sample context sentences more relevant to the span
    context_text = [
        "Natural language processing is a subfield of artificial intelligence.",
        "Python has become the dominant language for data science applications.",
        "Many developers use Python libraries like spaCy and NLTK for text processing.",
    ]

    # Parse with spaCy
    context_docs = [nlp(text) for text in context_text]

    # Target span
    target_doc = nlp("I like using Python for natural language processing.")
    target_span = target_doc[2:7]  # "using Python for natural language"

    # Initialize the similarity calculator
    sim_calc = SentenceSimilarityCalculator(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # Calculate similarity between concatenated context and target span
    similarity = sim_calc.calculate_similarity(context_docs, target_span)
    print(f"Similarity between concatenated context and target span: {similarity:.4f}")

    # Example with string input instead of spaCy objects
    similarity_str = sim_calc.calculate_similarity(context_text, target_span)
    print(f"Similarity using string input: {similarity_str:.4f}")
