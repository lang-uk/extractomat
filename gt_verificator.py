import csv
import argparse
from pathlib import Path
from matcha import ngrams
from collections import OrderedDict, Counter
from Levenshtein import jaro
import spacy
from spacy_layout import spaCyLayout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GT terms")
    parser.add_argument("text", type=Path, help="Path to text file (doc, pdf, txt)")
    parser.add_argument(
        "gt_path", type=Path, help="Path to ground truth CSV file with terms"
    )
    parser.add_argument("output_path", type=Path, help="Path to save the F1 curve plot")
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="Spacy model to use for tokenization and lemmatization",
    )
    parser.add_argument(
        "--full_gt_dataset",
        type=Path,
        help="Store all the found terms with repetitions in the order they were found",
    )

    args = parser.parse_args()

    nlp = spacy.load(args.model)
    layout = spaCyLayout(nlp)
    raw_doc = layout(args.text)

    text = nlp(raw_doc.text.lower())
    lengths = Counter()

    terms_arr = []
    max_len = 0
    with args.gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # Skip empty lines
                term = row[0].strip().lower()  # Case-insensitive matching
                terms_arr.append(term)
                lengths.update([term.count(" ") + 1])
                max_len = max(max_len, term.count(" ") + 1)

    terms_best_match = OrderedDict((term, 0) for term in terms_arr)
    terms_best_counterpart = OrderedDict((term, "") for term in terms_arr)

    print("Lengths of terms:")
    for length, count in lengths.most_common():
        print(f"Length {length}: {count} terms")

    print(f"max_len is {max_len}")

    with args.full_gt_dataset.open("w", encoding="utf-8") as f:
        writer = csv.writer(f)
        matches_in_text = list()

        for phrase in ngrams(text, n_min=1, n_max=max_len):
            phrase_str = (
                (" ".join(tok.text for tok in phrase)).lower().replace(" - ", "-")
            )
            phrase_lemma_str = (
                (" ".join(tok.lemma_ for tok in phrase)).lower().replace(" - ", "-")
            )

            for term in terms_arr:
                score_orig = jaro(phrase_str, term)
                score_lemma = jaro(phrase_lemma_str, term)

                if score_orig > terms_best_match[term]:
                    terms_best_match[term] = score_orig
                    terms_best_counterpart[term] = phrase_str

                if score_orig == 1.0:
                    writer.writerow([term])
                    matches_in_text.append(term)
                    break

                # if score_lemma > terms_best_match[term]:
                #     terms_best_match[term] = score_lemma
                #     terms_best_counterpart[term] = phrase_lemma_str

    exact_matches = 0
    partial_matches = 0
    em = []
    with args.output_path.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "score", "counterpart"])
        writer.writeheader()
        for term, score in terms_best_match.items():
            if score == 1.0:
                exact_matches += 1
                em.append(term)
            else:
                print(term, score)
                partial_matches += 1
            writer.writerow(
                {
                    "term": term,
                    "score": score,
                    "counterpart": terms_best_counterpart.get(term, ""),
                }
            )

    print(f"Exact matches: {exact_matches}")
    print(f"Partial matches: {partial_matches}")
    print(f"Matches in text: {len(matches_in_text)}")
    print(f"Total matches: {exact_matches + partial_matches}, {len(terms_arr)} terms")
