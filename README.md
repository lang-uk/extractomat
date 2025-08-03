# Extractomat: Automatic Term Extraction (ATE) for English/German and Ukrainian

## Features
1. Works on top of Spacy
2. Implements basic/combo-basic/c-value algorithms with extra flexibility and support for single word terms (see `matcha.py`)
3. Implements optional reranking using sentence transformers to weight the terms in the context of the document (see `sbert_reranker.py`)
4. Allows to run term extraction on txt/pdf/docx documents (see `runner.py`)
5. Comes with OTRT dataset (in English/German/Ukrainian)
6. Covered with tests.
7. Equipped with other experimental features (`keybert_extract.py`, `gliner_extract.py`) and scripts for measuring the performance on the ORTR dataset (`tester.py`, `gt_verificator.py`)


## Installation

```bash
# clone the repo
git clone https://github.com/lang-uk/extractomat
cd extractomat

# Activate virtual environment and install dependencies
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Download Spacy models for your language of interest
spacy download uk_core_news_trf
spacy download en_core_web_trf
spacy download de_dep_news_trf
```

## Running in the batch mode:
To run extractomat on the list of files, you can do:
```bash

python runner.py "my_papers_corpus/paper_*" --method cvalue --allow-single-word --n-max 6
```

Please consult with `python runner.py --help` for extra options.

## Running tests:
Just start `python -m pytest` and relax.

## OTRT dataset
The OTRT dataset (first page of the ONTOLOGIES OF TIME: REVIEW AND TRENDS paper) is in the `otrt` folder.

`gt_terms_*.csv` is the unique list of terms from the paper.
`gt_terms_*_full_ordered.csv` is the complete list of terms in the correct order (as they occur in the text)
`TimeOnto Sample *.docx` is the original text of the paper.


## Practical application
We used `extractomat` in our experiments on building unsupervised bilingual glossary. See https://github.com/lang-uk/schmezaurus for details.


## Citing the paper
Extractomat has been released as a part of the research results leading to the following paper: 

Chaplynskyi, D., Wittenborg, T., Kosa, V., Rabby, G., Ignatenko, O., Auer, S., Ermolayev, V. (2025) Building Multilingual Terminological Bridges between Language-Specific Knowledge Silos. In: Proc. ICTERI-2025, CCIS, Springer (to appear)