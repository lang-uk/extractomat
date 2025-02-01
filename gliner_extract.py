"""GLiNER term extraction module."""

import argparse
from typing import Dict, Set, Tuple, Union, List, Optional
import spacy
from spacy.tokens import Doc
import torch
from collections import defaultdict


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def create_spacy_pipeline(
    model: str,
    base_model: str = "en_core_web_sm",
    labels: Optional[List[str]] = None,
    style: str = "ent",
    chunk_size: int = 250,
    threshold: float = 0.0,
) -> spacy.language.Language:
    """Create a spaCy pipeline with GLiNER configuration."""
    if style not in ["ent", "span"]:
        raise ValueError("Style must be either 'ent' or 'span'")

    if labels is None:
        labels = ["term"]

    config = {
        "gliner_model": model,
        "chunk_size": chunk_size,
        "labels": labels,
        "style": style,
        "threshold": threshold,
    }

    nlp = spacy.load(base_model, disable=["parser", "entity"])
    nlp.add_pipe("gliner_spacy", config=config)
    return nlp


def merge_entities(doc: Doc) -> List[spacy.tokens.Span]:
    """Merge adjacent entities with same label.

    Args:
        doc: Processed spaCy Doc with entities

    Returns:
        List of merged entity spans
    """
    if not doc.ents:
        return []

    merged = []
    current = doc.ents[0]

    for next_ent in doc.ents[1:]:
        if next_ent.label_ == current.label_ and (
            next_ent.start == current.end or next_ent.start == current.end + 1
        ):
            # Create new span combining current and next
            current = doc[current.start : next_ent.end]
            # Copy score from first entity
            current._.score = doc.ents[0]._.score
        else:
            merged.append(current)
            current = next_ent

    merged.append(current)
    return merged


def extract_terms(
    doc: Union[str, Doc], nlp: spacy.language.Language, merge: bool = False
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """Extract terms from text using GLiNER model."""
    if isinstance(doc, str):
        doc = nlp(doc)
    elif not isinstance(doc, Doc):
        raise TypeError("doc must be either str or spacy.tokens.Doc")

    term_scores: Dict[str, float] = {}
    term_occurrences: Dict[str, Set[str]] = defaultdict(set)

    entities = merge_entities(doc) if merge else doc.ents

    for ent in entities:
        lemma = " ".join([token.lemma_.lower() for token in ent])
        current_score = term_scores.get(lemma, 0.0)
        term_scores[lemma] = max(current_score, ent._.score)
        term_occurrences[lemma].add(ent.text)

    return term_scores, dict(term_occurrences)


def gliner_extract(
    doc: Union[str, Doc],
    model: str,
    labels: Optional[List[str]] = None,
    style: str = "ent",
    threshold: float = 0.0,
    merge: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """Main function for GLiNER-based term extraction."""
    if labels and not all(label.islower() for label in labels):
        raise ValueError("All labels must be lowercase")

    nlp = create_spacy_pipeline(model=model, labels=labels, style=style, threshold=threshold)

    return extract_terms(doc, nlp, merge)


if __name__ == "__main__":  # pragma: no cover
    text = """Ontologies of Time: Review and Trends
    Time, as a phenomenon, has been in the focus of scientific thought from ancient times. It continues to be an important subject of research in many disciplines due to its importance as a basic aspect for understanding and formally representing change. The goal of this analytical review is to find out if the formal representations of time developed to date suffice to the needs of the basic and applied research in Computer Science, and in particular within the Artificial Intelligence and Semantic Web communities. To analyze if the existing basic theories, models, and implemented ontologies of time cover these needs well, the set of the features of time has been extracted and appropriately structured using the paper collection of the TIME Symposia series as the document corpus. This feature set further helped to structure the comparative review and analysis of the most prominent temporal theories. As a result, the selection of the subset of the features of time (the requirements for a Synthetic Theory) has been made reflecting the TIME community sentiment.  Further, the temporal logics, representation languages, and ontologies available to date, have been reviewed regarding their usability aspects and the coverage of the selected temporal features. The results reveal that the reviewed ontologies of time taken together do not satisfactorily cover some important features: (i) density; (ii) relaxed linearity; (iii) scale factors; (iv) proper and periodic subintervals; (v) temporal measures and clocks.  It has been concluded that a cross-disciplinary effort is required to address the features not covered by the existing ontologies of time, and also harmonize the representations addressed differently.   
    Keywords: Time; sentiment; temporal feature; coverage; ontology; representation; reasoning.
    Introduction
    It is acknowledged that “when God made time, he made plenty of it”. Remarkably, when it goes about the formal treatment of time, the status is very much following this Irish saying.  Time, as a phenomenon, has been in the focus of scientific thought from ancient times. Today it continues to be an important subject of research for philosophers, physicists, mathematicians, logicians, computer scientists, and even biologists. One reason, perhaps, is that time is a fundamental aspect to understand and react to change in the World, including the broadest diversity of applications that impact the evolution of the Humankind. So, the progress in understanding the World in its dynamics: (a) is based on having an adequately rich and deep model of time; and (b) pushes forward the further refinement of our time models.  For example, in Computer Science the developments in Artificial Intelligence, Databases, Distributed Systems, etc. in the last two decades have brought to life several prominent theoretical frameworks dealing with temporal aspects. Some parts of these theories gave boost to the research in logics – yielding a family of temporal logics, comprising temporal description logics. Based on this logical foundation, knowledge representation languages have received their capability to represent time, and several ontologies of time have been implemented by the Semantic Web community.  It is however important to find out if this plenty is enough to meet the requirement in Computer Science research and development.
    The objective of this analytic review paper is to answer this question – i.e. to find out if the formal representations of time developed to date suffice to the needs of coping with different aspects of change. The remainder of the paper is structured as follows. 
    """

    parser = argparse.ArgumentParser(description="GLiNER term extraction")
    parser.add_argument("--text", type=str, help="Input text", default=text)
    parser.add_argument("model", type=str, help="GLiNER model name or path")
    parser.add_argument("--labels", type=str, nargs="+", help="Entity labels to extract")
    parser.add_argument("--style", type=str, default="ent", help="Output style ('ent' or 'span')")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Score threshold for entity detection"
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge adjacent entities with same label"
    )
    args = parser.parse_args()

    term_scores, term_occurrences = gliner_extract(
        doc=args.text,
        model=args.model,
        labels=args.labels,
        style=args.style,
        threshold=args.threshold,
        merge=args.merge,
    )

    from pprint import pprint

    pprint({"term_scores": term_scores, "term_occurrences": term_occurrences})
