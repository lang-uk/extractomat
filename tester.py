import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, List
import csv

import spacy
from spacy.tokens import Span
from spacy_layout import spaCyLayout
import numpy as np
import matplotlib.pyplot as plt

from matcha import (
    combo_basic,
    basic,
    cvalue,
)


class TermEvaluator:
    """Evaluates term extraction against ground truth."""

    def __init__(
        self,
        gt_path: Path,
        term_scores: Dict[str, float],
        term_occurrences: Dict[str, List[Span]],
        filter_single_word: bool = True,
    ):
        """
        Args:
            gt_path: Path to ground truth CSV
            term_scores: Mapping of lemmatized terms to their scores
            term_occurrences: Mapping of lemmatized terms to their occurrences
            filter_single_word: Whether to filter out single-word terms from GT
        """
        self.term_scores = term_scores
        self.term_occurrences = {
            lemma: set(term.text.lower() for term in terms)
            for lemma, terms in term_occurrences.items()
        }
        self.gt_terms = self._load_gt_terms(gt_path, filter_single_word)

    def _load_gt_terms(self, path: Path, filter_single_word: bool) -> Set[str]:
        """Load ground truth terms from CSV."""
        terms = set()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():  # Skip empty lines
                    term = row[0].strip().lower()  # Case-insensitive matching
                    if term.startswith("//"):  # Skip comments
                        continue

                    if not filter_single_word or len(term.split()) > 1:
                        terms.add(term)
        return terms

    def _is_term_match(self, gt_term: str, lemma: str) -> bool:
        """
        Check if ground truth term matches any form of the extracted term.

        Args:
            gt_term: Ground truth term (already lowercase)
            lemma: Lemmatized form of extracted term

        Returns:
            True if there's a match
        """
        if lemma.lower() == gt_term:
            return True

        occurrences = self.term_occurrences.get(lemma, set())
        return any(occ.lower() == gt_term for occ in occurrences)

    def calculate_metrics(
        self, threshold: float, verbose: bool = False
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall and F1 score at given threshold.

        Args:
            threshold: Score threshold for including terms
            verbose: Whether to print unmatched terms

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Filter terms by threshold
        filtered_terms = {
            term: score
            for term, score in self.term_scores.items()
            if score >= threshold
        }

        if not filtered_terms:
            return 0.0, 0.0, 0.0

        # Count matches and track unmatched terms
        true_positives = 0
        matched_gt_terms = set()
        matched_extracted_terms = set()

        for gt_term in self.gt_terms:
            for lemma in filtered_terms:
                if self._is_term_match(gt_term, lemma):
                    true_positives += 1
                    matched_gt_terms.add(gt_term)
                    matched_extracted_terms.add(lemma)
                    break

        if verbose:
            # Print unmatched terms
            unmatched_gt = self.gt_terms - matched_gt_terms
            unmatched_extracted = set(filtered_terms.keys()) - matched_extracted_terms

            print("Unmatched ground truth terms:", sorted(unmatched_gt))
            print("Unmatched extracted terms:", sorted(unmatched_extracted))

        precision = true_positives / len(filtered_terms)
        recall = true_positives / len(self.gt_terms)

        if precision + recall == 0:
            return 0.0, 0.0, 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def plot_f1_curve(
        self,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
        steps: int = 50,
        n_bins: int = 30,
        output_path: Path = None,
    ) -> None:
        """
        Plot F1 scores over different thresholds with score distribution histogram.

        Args:
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            steps: Number of threshold points to evaluate
            n_bins: Number of bins for the histogram
            output_path: Optional path to save the plot
        """
        thresholds = np.linspace(min_threshold, max_threshold, steps)
        metrics = [self.calculate_metrics(t) for t in thresholds]

        precisions, recalls, f1_scores = zip(*metrics)

        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True
        )
        fig.subplots_adjust(hspace=0.1)  # Reduce space between plots

        # Plot F1 curves on top subplot
        ax1.plot(thresholds, precisions, label="Precision", linestyle="--")
        ax1.plot(thresholds, recalls, label="Recall", linestyle="--")
        ax1.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)

        ax1.set_ylabel("Score")
        ax1.set_title("Term Extraction Performance vs Threshold")
        ax1.legend()
        ax1.grid(True)

        # Find and mark optimal F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_recall = recalls[best_idx]
        best_precision = precisions[best_idx]

        ax1.plot(best_threshold, best_f1, "ro")
        ax1.annotate(
            f"Best F1: {best_f1:.3f}\nThreshold: {best_threshold:.3f}",
            (best_threshold, best_f1),
            xytext=(10, 10),
            textcoords="offset points",
        )
        print(
            f"Best F1: {best_f1:.3f} (p: {best_precision:.3f} / r: {best_recall:.3f}) at threshold {best_threshold:.3f}"
        )

        # Calculate optimal number of bins
        scores = list(self.term_scores.values())

        if n_bins is None:  # Use Freedman-Diaconis rule by default
            n = len(scores)
            scores_array = np.array(scores)
            iqr = np.percentile(scores_array, 75) - np.percentile(scores_array, 25)
            bin_width = 2 * iqr * (n ** (-1 / 3)) if iqr > 0 else 0.1
            n_bins = max(int((max(scores) - min(scores)) / bin_width), 1)

        # Plot histogram of scores on bottom subplot
        ax2.hist(scores, bins=n_bins, edgecolor="black")
        ax2.set_xlabel("Score Threshold")
        ax2.set_ylabel("Count")
        ax2.grid(True)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        plt.show()


def mimic_term_scores(terms):
    # [
    #     "Ontologies",
    #     "understanding",
    #     "representations",
    #     "Computer Science",
    #     "Artificial Intelligence",
    #     "Semantic Web",
    #     "theories",
    #     "ontologies",
    #     "document",
    #     "feature set",
    #     "theories",
    #     "Theory",
    #     "community",
    #     "sentiment",
    #     "ontologies",
    #     "usability",
    #     "temporal features",
    #     "ontologies",
    #     "measures",
    #     "ontologies",
    #     "representations",
    #     "sentiment",
    #     "temporal feature",
    #     "ontology",
    #     "reasoning",
    #     "understanding",
    #     "Computer Science",
    #     "Artificial Intelligence",
    #     "Databases",
    #     "theories",
    #     "temporal description",
    #     "knowledge representation",
    #     "ontologies",
    #     "Semantic Web",
    #     "community",
    #     "Computer Science research and development",
    #     "representations",
    # ]

    return {term: 1.0 for term in terms}, {term: set() for term in terms}


if __name__ == "__main__":
    # text = """Ontologies of Time: Review and Trends
    # Time, as a phenomenon, has been in the focus of scientific thought from ancient times. It continues to be an important subject of research in many disciplines due to its importance as a basic aspect for understanding and formally representing change. The goal of this analytical review is to find out if the formal representations of time developed to date suffice to the needs of the basic and applied research in Computer Science, and in particular within the Artificial Intelligence and Semantic Web communities. To analyze if the existing basic theories, models, and implemented ontologies of time cover these needs well, the set of the features of time has been extracted and appropriately structured using the paper collection of the TIME Symposia series as the document corpus. This feature set further helped to structure the comparative review and analysis of the most prominent temporal theories. As a result, the selection of the subset of the features of time (the requirements for a Synthetic Theory) has been made reflecting the TIME community sentiment.  Further, the temporal logics, representation languages, and ontologies available to date, have been reviewed regarding their usability aspects and the coverage of the selected temporal features. The results reveal that the reviewed ontologies of time taken together do not satisfactorily cover some important features: (i) density; (ii) relaxed linearity; (iii) scale factors; (iv) proper and periodic subintervals; (v) temporal measures and clocks.  It has been concluded that a cross-disciplinary effort is required to address the features not covered by the existing ontologies of time, and also harmonize the representations addressed differently.
    # Keywords: Time; sentiment; temporal feature; coverage; ontology; representation; reasoning.
    # Introduction
    # It is acknowledged that “when God made time, he made plenty of it”. Remarkably, when it goes about the formal treatment of time, the status is very much following this Irish saying.  Time, as a phenomenon, has been in the focus of scientific thought from ancient times. Today it continues to be an important subject of research for philosophers, physicists, mathematicians, logicians, computer scientists, and even biologists. One reason, perhaps, is that time is a fundamental aspect to understand and react to change in the World, including the broadest diversity of applications that impact the evolution of the Humankind. So, the progress in understanding the World in its dynamics: (a) is based on having an adequately rich and deep model of time; and (b) pushes forward the further refinement of our time models.  For example, in Computer Science the developments in Artificial Intelligence, Databases, Distributed Systems, etc. in the last two decades have brought to life several prominent theoretical frameworks dealing with temporal aspects. Some parts of these theories gave boost to the research in logics – yielding a family of temporal logics, comprising temporal description logics. Based on this logical foundation, knowledge representation languages have received their capability to represent time, and several ontologies of time have been implemented by the Semantic Web community.  It is however important to find out if this plenty is enough to meet the requirement in Computer Science research and development.
    # The objective of this analytic review paper is to answer this question – i.e. to find out if the formal representations of time developed to date suffice to the needs of coping with different aspects of change. The remainder of the paper is structured as follows.
    # """.lower()

    parser = argparse.ArgumentParser(
        description="Evaluate term extraction against ground truth"
    )
    parser.add_argument("text", type=Path, help="Path to text file (doc, pdf, txt)")
    parser.add_argument(
        "gt_path", type=Path, help="Path to ground truth CSV file with terms"
    )
    parser.add_argument("output_path", type=Path, help="Path to save the F1 curve plot")
    parser.add_argument(
        "--method",
        type=str,
        choices=["basic", "cvalue", "combo_basic"],
        default="basic",
        help="Term extraction method to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
    )
    parser.add_argument("--allow-single-word", default=False, action="store_true")

    args = parser.parse_args()

    nlp = spacy.load(args.model, disable=["parser", "entity"])
    layout = spaCyLayout(nlp)
    raw_doc = layout(args.text)
    tagged_doc = nlp(raw_doc.text.lower())

    n_min = 1 if args.allow_single_word else 2

    if args.method == "basic":
        term_scores, term_occurrences = basic(tagged_doc, n_min=n_min)
    elif args.method == "cvalue":
        smoothing = 1 if args.allow_single_word else 0.1
        term_scores, term_occurrences = cvalue(
            tagged_doc, n_min=n_min, smoothing=smoothing, n_max=8
        )
    elif args.method == "combo_basic":
        term_scores, term_occurrences = combo_basic(tagged_doc, n_min=n_min)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    if not term_scores:
        print("No terms extracted")
        exit()

    terms_max_score = max(term_scores.values())
    terms_min_score = min(term_scores.values())

    evaluator = TermEvaluator(
        gt_path=args.gt_path,
        term_scores=term_scores,
        term_occurrences=term_occurrences,
        filter_single_word=not args.allow_single_word,
    )

    print(evaluator.calculate_metrics(0.0, verbose=True))

    evaluator.plot_f1_curve(
        min_threshold=terms_min_score,
        max_threshold=terms_max_score,
        steps=min(len(term_scores), 50),
        output_path=args.output_path,
        n_bins=None,
    )
