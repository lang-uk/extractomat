import json
from typing import List, Union, Optional, Dict
from collections import defaultdict
from pathlib import Path
from statistics import median

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
        self.score_per_length = defaultdict(list)
        self.adjusted_score_per_length = defaultdict(list)
        self.score_per_term = defaultdict(list)
        self.adjusted_score_per_term = defaultdict(list)

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
    ) -> float:
        """
        Calculate cosine similarity between concatenated context sentences and the target span,
        with optional non-linear adjustment for shorter spans.

        Args:
            context_sentences: List of sentences to concatenate and compare against (as strings or spaCy objects)
            target_span: spaCy Span to compare with the concatenated context

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
        self.score_per_length[len(target_span)].append(
            base_similarity
        )
        self.score_per_term[target_span.text.lower()].append(base_similarity)

        return base_similarity

    def rerank_terms_in_doc(
        self,
        doc: Doc,
        term_occurences=Dict[str, List[Span]],
        context_len: int = 3,
        pooling="max",
        length_adjustment: str = "none",
        legacy_length_factor: float = 0.2,
        legacy_min_tokens: int = 3,
        legacy_max_boost: float = 0.5,
    ) -> Dict[str, float]:
        """
        Rerank terms in a document based on their similarity to the context.

        Args:
            doc: spaCy Doc object
            term_occurences: Dictionary of term occurences (spans) in the document
            context_len: Number of context sentences to consider
            pooling: Pooling strategy for term embeddings (max, mean). If there
            is more than one occurence of a term, the similarity scores for it
                pooled from all occurences using the specified strategy.
            length_adjustment: Length adjustment strategy to boost short terms scores
                (none, legacy, median, modified_z_score)

            legacy_length_factor: Controls the strength of the length adjustment
                (higher = more adjustment)
            legacy_min_tokens: Minimum number of tokens that can receive adjustment
            legacy_max_boost: Maximum boost that can be applied to the similarity score
        Returns:
            Lemma-score dict as Dict[str, float]
        """
        assert pooling in ["max", "mean"], "Pooling must be 'max' or 'mean'"
        assert length_adjustment in ["none", "legacy", "median", "modified_z_score"], (
            "Length adjustment must be 'none', 'legacy', 'median', or 'modified_z_score'"
        )

        # Extract context sentences
        context_sentences = extract_context(doc, term_occurences, context_len)

        # Calculate similarity for each term and group them by lemma and occurence (span)
        term_scores = defaultdict(lambda: defaultdict(list))

        for lemma, occurences in context_sentences.items():
            for context, span in occurences:
                term_scores[lemma][span].append(
                    self.calculate_similarity(
                        context_sentences=context, target_span=span
                    )
                )

        medians = {k: median(v) for k, v in self.score_per_length.items()}
        max_median = max(medians.values())
        medians_factor = {k: max_median / v for k, v in medians.items()}

        mads = {}
        for k, k_median in medians.items():
            mads[k] = median([abs(v - k_median) for v in self.score_per_length[k]])

        # Apply length adjustment to span scores if specified
        for lemma, occ_scores in term_scores.items():
            for term, scores in occ_scores.items():
                term_length = len(term)

                adjusted_scores = []

                for score in scores:
                    if length_adjustment == "none":
                        adjusted_scores.append(score)
                    elif length_adjustment == "legacy":
                        # Calculate a boosting factor based on span length
                        # The shorter the span, the more boost (with diminishing returns)

                        if term_length <= legacy_min_tokens:
                            length_boost = legacy_max_boost * np.exp(-legacy_length_factor * term_length)

                            # Apply the boost, ensuring we don't exceed 1.0
                            adjusted_scores.append(min(1.0, score + length_boost))
                        else:
                            adjusted_scores.append(score)
                    elif length_adjustment == "median":
                        adjusted_scores.append(
                            score * medians_factor[term_length]
                        )
                    elif length_adjustment == "modified_z_score":
                        adjusted_scores.append(
                            0.6745 * (score - medians[term_length]) / mads[term_length]
                        )

                self.adjusted_score_per_length[len(term)] += adjusted_scores
                self.adjusted_score_per_term[str(term)] += adjusted_scores

                occ_scores[term] = adjusted_scores

        # Calculate final scores for each lemma based on the specified pooling strategy
        # and the adjusted scores of individual occurences of that lemma

        final_lemma_scores = {}
        for lemma, occurences in term_scores.items():
            all_lemma_scores = []
            for term, scores in occurences.items():
                all_lemma_scores.extend(scores)

            # Pool scores based on the specified strategy
            if pooling == "max":
                final_lemma_scores[lemma] = max(all_lemma_scores)
            else:
                final_lemma_scores[lemma] = sum(all_lemma_scores) / len(all_lemma_scores)

        return final_lemma_scores

    def export_score_per_length(self, filename: Path):
        """
        Export the score per length to a file.

        Args:
            filename: Name of the file to save the scores
        """

        with filename.open("w", encoding="utf-8") as f:
            json.dump(self.score_per_length, f, indent=4)

        with filename.with_stem(filename.stem + "_adjusted").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(self.adjusted_score_per_length, f, indent=4)


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
