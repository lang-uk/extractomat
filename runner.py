import argparse
from glob import glob
from pathlib import Path
import csv

from tqdm import tqdm
import spacy
from spacy_layout import spaCyLayout

from matcha import (
    combo_basic,
    basic,
    cvalue,
)

from sbert_reranker import SentenceSimilarityCalculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Matcha term extraction.")
    parser.add_argument("input_glob", type=str, help="Glob pattern for input files.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["basic", "combo_basic", "cvalue", "rerank"],
        default="basic",
        help="Method to use for the extraction.",
    )

    parser.add_argument(
        "--rerank-score-adjustment",
        type=str,
        choices=["none", "modified_z_score", "median", "legacy"],
        default="modified_z_score",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_trf",
    )
    parser.add_argument("--allow-single-word", default=False, action="store_true")
    parser.add_argument(
        "--n-max", type=int, default=6, help="Maximum length of terms to extract."
    )
    args = parser.parse_args()

    nlp = spacy.load(
        args.model,
        disable=[
            "entity",
        ],
    )
    nlp.max_length = 10_000_000
    layout = spaCyLayout(nlp)

    n_min = 1 if args.allow_single_word else 2

    for input_file in tqdm(list(map(Path, glob(args.input_glob)))):
        output_file = input_file.with_suffix(f".{args.method}.csv")
        if output_file.exists():
            print(f"Skipping {input_file} as {output_file} already exists.")
            continue

        if input_file.suffix == ".txt":
            tagged_doc = nlp(input_file.read_text(encoding="utf-8").lower())
        if input_file.suffix == ".csv":
            print(f"Skipping {input_file} as CSV input is not supported.")
            continue
        else:
            raw_doc = layout(input_file)
            tagged_doc = nlp(raw_doc.text.lower())

        if args.method == "basic":
            term_scores, term_occurrences = basic(tagged_doc, n_min=n_min)
        elif args.method in ["cvalue", "rerank"]:
            smoothing = 1 if args.allow_single_word else 0.1
            term_scores, term_occurrences = cvalue(
                tagged_doc, n_min=n_min, smoothing=smoothing, n_max=4
            )

            if args.method == "rerank":
                reranker = SentenceSimilarityCalculator(
                    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                )

                term_scores = reranker.rerank_terms_in_doc(
                    tagged_doc,
                    term_occurrences,
                    context_len=3,
                    pooling="max",
                    length_adjustment=args.rerank_score_adjustment,
                )

        elif args.method == "combo_basic":
            term_scores, term_occurrences = combo_basic(
                tagged_doc, n_min=n_min, n_max=args.n_max
            )
        else:
            raise ValueError(f"Invalid method: {args.method}")

        with open(
            output_file, "w", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["term", "score", "occurrences"],
            )

            writer.writeheader()
            for term, score in term_scores.items():
                writer.writerow(
                    {
                        "term": term,
                        "score": score,
                        "occurrences": ", ".join(
                            set(
                                occ.text.lower()
                                for occ in term_occurrences.get(term, [])
                            )
                        ),
                    }
                )
