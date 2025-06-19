from typing import Iterator, Optional, Tuple, Dict, Set, List
import re
import math
from collections import defaultdict, Counter
from tabulate import tabulate

import spacy
from spacy.tokens import Doc, Span


# This regex is based on the pattern (NN(S)?_|JJ_|NNP_|NN(S?)_IN_)*(NN(S)?)
# found in the https://github.com/ispras/atr4s repo
# SECRET_REGEX = re.compile(r"^(?!HYPH_)(?!CC_)(NNS?_|JJ_|HYPH_|CC_|NNP_|NNS?_IN_)*(NNS?_)$")
# SECRET_REGEX = re.compile(r"^(?!HYPH_)(NNS?_|JJS?_|HYPH_|NNPS?_|NNS?_IN_)*(NN[PS]*_)$")

# Pattern for multiple word term:
#   * Not starting with HYPH
#   * Any numbers of any of these
#       * Nouns (singular, plural or proper) or
#       * Adjective (usual or superlative) or
#       * noun + preposition
#       * Maybe a hyph (part of a compound)
#   * Ending with a noun (singular or plural)
SECRET_REGEX = re.compile(r"^(?!HYPH_)(NNS?_|JJS?_|HYPH_|NNPS?_|NNS?_IN_)*(NNS?_)$")

# Pattern for one word:
#   * noun (singular or plural) or // should we include also proper nouns?
#   * adjective (also comparative or superlative) // probably bad idea, disabled for now
SECRET_REGEX_FOR_ONE_WORD = re.compile(r"^(NNS?_)$")

# UPOS to Penn mapping dictionary
UPOS_TO_PENN: Dict[str, str] = {
    "NOUN": "NN",  # Default to singular
    "PROPN": "NNP",  # Default to singular
    "ADJ": "JJ",  # Default to basic form
    "ADP": "IN",
    "ADJA": "JJ",       # German
    "KOUS": "IN",       # German
    "KON": "CC",        # German
    "APPR": "IN",       # German
    "APPRART": "IN",    # German
    "KOUI": "IN",       # German
    "VERB": "VB",       # Dutch
    "AUX": "MD",        # Dutch
    "ADV": "RB",        # Dutch
    "PRON": "PRP",      # Dutch
    "DET": "DT",        # Dutch
    "CCONJ": "CC",      # Dutch
    "SCONJ": "IN",      # Dutch
    "PART": "RP",       # Dutch
    "INTJ": "UH",       # Dutch
    "NUM": "CD",        # Dutch
    "SYM": "SYM",       # Dutch
    "PUNCT": ".",       # Dutch
    "X": "NN",          # Dutch
    "SPACE": "NN",      # Dutch
}


def upos_to_penn(tags: List[str]) -> List[str]:
    """Convert Universal POS tags to Penn Treebank tags.

    Args:
        tags: List of Universal POS tags to convert

    Returns:
        List of converted Penn Treebank tags. Tags that cannot be
        converted are left unchanged.
    """

    return [UPOS_TO_PENN.get(tag, tag) for tag in tags]


def ngrams(
    document: Doc | List[str], n_min: int = 1, n_max: Optional[int] = None
) -> Iterator[Span | List[str]]:
    """Yields ngrams of variable length from a spaCy Doc object.

    Args:
        doc: spaCy Doc object to process
        n_min: Minimum length of ngrams (default=1)
        n_max: Maximum length of ngrams (default=None, meaning use doc length)

    Returns:
        Iterator yielding spaCy Spans of varying lengths

    Raises:
        ValueError: If n_min < 1 or n_max < n_min
    """
    if n_min < 1:
        raise ValueError("n_min must be at least 1")

    # If n_max not specified, use document length
    if n_max is None:
        n_max = len(document)

    if n_max < n_min:
        raise ValueError("n_max must be greater than or equal to n_min")

    # For each possible ngram length
    for n in range(n_min, n_max + 1):
        # For each possible starting position
        for start_idx in range(len(document) - (n - 1)):
            yield document[start_idx : start_idx + n]


def extract_context(
    doc: Doc, lemma_spans: Dict[str, List[Span]], n: int = 1
) -> Dict[str, List[Tuple[List[Span], Span]]]:
    """
    Extract n sentences surrounding each span in the lemma_spans mapping.

    Args:
        doc: A spaCy Doc object
        lemma_spans: A dictionary mapping lemmas to lists of Span objects
        n: Number of sentences to extract (n=1: only the sentence where span belongs,
                                          n=2: span's sentence + one before,
                                          n=3: span's sentence + one before and after, etc.)

    Returns:
        A dictionary mapping lemmas to lists of tuples, where each tuple contains:
        - A list of sentence Spans (the surrounding sentences)
        - The original Span
    """
    # Get all sentences in the document
    sentences = list(doc.sents)
    total_sentences = len(sentences)

    result = {}

    for lemma, spans in lemma_spans.items():
        result[lemma] = []

        for span in spans:
            # Find all sentences that overlap with the span
            overlapping_indices = []
            for i, sent in enumerate(sentences):
                if span.start < sent.end and span.end > sent.start:
                    overlapping_indices.append(i)

            if not overlapping_indices:
                continue  # Skip if span doesn't overlap with any sentence

            # Find the first and last sentence indices containing the span
            min_idx = min(overlapping_indices)
            max_idx = max(overlapping_indices)

            start_idx = min_idx
            end_idx = max_idx

            if n > 1:
                # We need to add (n - 1) more sentences
                remaining = n - 1

                # Add sentences before the span first
                before_count = min(min_idx, math.ceil(remaining / 2))
                start_idx = min_idx - before_count

                # Then add sentences after the span if needed
                after_count = min(
                    total_sentences - 1 - max_idx, remaining - before_count
                )
                end_idx = max_idx + after_count

            # Get the sentence spans
            context_sentences = [sentences[i] for i in range(start_idx, end_idx + 1)]

            # Add to result
            result[lemma].append((context_sentences, span))

    return result


def subphrases(phrase: str | Doc | Span, n_min: int = 2) -> Iterator[str]:
    """
    Generate all possible contiguous subsequences of a tokenized phrase,
    except the full phrase itself.

    Args:
        phrase: str, spaCy Doc or Span object
        n_min: Minimum length of the substring in words (default=2)
    Returns:
        Iterator[str]: Iterator yielding all possible contiguous subsequences
    """

    if isinstance(phrase, str):
        tokens = phrase.split()
    else:
        tokens = [token.text for token in phrase]

    if len(tokens) < n_min + 1:
        return

    for subs in ngrams(tokens, n_min=n_min, n_max=len(tokens) - 1):
        yield " ".join(subs)


def get_pos_fingerprint(phrase: Span) -> str:
    """Generate a POS fingerprint for a spaCy Span.

    Args:
        phrase: spaCy Span object

    Returns:
        String representing the POS fingerprint
    """

    for token in phrase:
        if token.text in ["-", "–", "—", "−"]:
            token.tag_ = "HYPH"
        elif token.is_punct:
            token.tag_ = "PUNCT"

    converted_tags = upos_to_penn([token.tag_ for token in phrase])
    return "".join(token_tag + "_" for token_tag in converted_tags)


def get_lemmatized_phrase(phrase: Span) -> List[str]:
    """Return a lemmatized version of the phrase.

    Args:
        phrase: spaCy Span object

    Returns:
        Tokenized representation the lemmatized phrase
    """
    return [token.lemma_.lower() for token in phrase]


def is_phrase_matching(phrase: Span, allow_single_word: bool = False) -> bool:
    """
    Determine if a phrase is good for us

    Args:
        phrase: spaCy Span object
        allow_single_word: If True, allow single word phrases (default=False)
    Returns:
        bool: True if the phrase is matching the secret regex
    """
    phrase_fingerprint = get_pos_fingerprint(phrase)
    # print(phrase.text, phrase_fingerprint)

    # To cover cases when HYPH is recognized as JJ (part of a compound)
    if phrase.text.startswith("-"):
        return False

    if allow_single_word:
        if SECRET_REGEX_FOR_ONE_WORD.match(phrase_fingerprint):
            return True

    # if phrase.text.lower() in [
    #     "applied research",
    #     "basic and applied research",
    #     "comparative review and analysis",
    #     "computer science research and development",
    #     "distributed systems",
    #     "god",
    #     "most prominent temporal theory",
    #     "progress in understanding the world",
    #     "rich and deep model of time",
    #     "set of the features of time",
    #     "subset of the features of time",
    #     "understanding",
    #     # Ukrainian:
    #     "фундаментальних і прикладних досліджень",
    #     "semantic web",
    #     "time",
    #     "адекватно насиченої і глибокої моделі часу",
    #     "насиченої і глибокої моделі часу",
    #     "порівняльний огляд та аналіз",
    #     "сентимент спільноти time",
    #     "серії симпозіумів time",
    #     "спільнот семантичного вебу",
    #     "спільнота semantic web",
    #     "спільноти time",
    #     "фундаментальних і прикладних досліджень",
    #     "synthetische theorie",
    # ]:
    #     print(phrase.text, phrase_fingerprint)

    # if phrase.text.lower().startswith("implemented "):
    #     print(phrase.text, phrase_fingerprint, get_lemmatized_phrase(phrase))

    # return True
    return bool(SECRET_REGEX.match(phrase_fingerprint))


def extract_terms(
    document: Doc, n_min: int = 2, n_max: int = 6, stopwords: Optional[Set] = None
) -> Tuple[Dict[str, int], Dict[str, List[Span]]]:
    """
    Process a document and return the term candidates as a lemma to frequency mapping
    and a mapping of lemma to occurences in the document

    Args:
        document: spaCy Doc object
        n_min: Minimum length of ngrams (default=2)
        n_max: Maximum length of ngrams (default=6)
        stopwords: Set of stopwords to ignore (default=None). Has to be lowercased
    Returns:
        Tuple[Dict[str, int], Dict[str, List[Span]]]: A tuple containing the lemma
            to frequency mapping and the lemma to occurences mapping
    """

    term_freq = Counter()
    phrase_occurrences = defaultdict(list)
    if n_min == 1:
        matcher = lambda x: is_phrase_matching(phrase=x, allow_single_word=True)
    else:
        matcher = is_phrase_matching

    for phrase in filter(matcher, ngrams(document, n_min=n_min, n_max=n_max)):
        lemmatized = get_lemmatized_phrase(phrase)
        lemmatized_str = " ".join(lemmatized)

        # Skip if any lemmatized token is in stopwords
        if stopwords and any(token in stopwords for token in lemmatized):
            continue

        term_freq[lemmatized_str] += 1
        phrase_occurrences[lemmatized_str].append(phrase)

    return term_freq, phrase_occurrences


def calculate_nesting(
    term_freqs: Dict[str, int], use_frequencies: bool = False
) -> Tuple[defaultdict[str, int], defaultdict[str, int]]:
    """Calculate nesting metrics for each term.

    For each term calculates:
    1. Number of times this term appears as part of other terms
    2. Number of times other terms appear as part of this term

    Args:
        term_freqs: Dictionary mapping terms to their frequencies
        use_frequencies: If True, multiply counts by term frequencies

    Returns:
        Tuple containing:
        - defaultdict mapping terms to their superset counts (e_t)
        - defaultdict mapping terms to their subset counts (e'_t)
    """
    superset_counts = defaultdict(int)  # e_t - how many terms contain this one
    subset_counts = defaultdict(int)  # e'_t - how many terms this one contains

    # For each term and its frequency
    for container_term, container_freq in term_freqs.items():
        # Get all potential subterms
        for subterm in subphrases(container_term):
            # If subterm is a valid term
            if subterm in term_freqs:
                if use_frequencies:
                    weight = container_freq
                else:
                    weight = 1

                superset_counts[subterm] += weight  # subterm appears in container_term
                subset_counts[
                    container_term
                ] += weight  # container_term contains subterm

    return superset_counts, subset_counts


def combo_basic(
    doc: Doc,
    alpha: float = 0.75,
    beta: float = 0.1,
    n_min: int = 2,
    n_max: int = 6,
    stopwords: Optional[Set[str]] = None,
    smoothing: float = 0,
    use_frequencies: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """Calculate ComboBasic score for term candidates.

    ComboBasic(t) = |t| * log(f(t)) + α*e_t + β*e'_t
    where:
        |t| is the length of term (number of words)
        f(t) is the frequency of the term
        e_t is the number of terms that contain t
        e'_t is the number of terms contained in t
        α, β are weight parameters

    Args:
        doc: spaCy Doc object
        alpha: Weight for e_t (default: 0.75)
        beta: Weight for e'_t (default: 0.1)
        n_min: Minimum length of terms in tokens (default: 2)
        n_max: Maximum length of terms in tokens (default: 6)
        stopwords: Optional set of stopwords to ignore (default: None)
        smoothing: Smoothing factor for log(f(t)) (default: 0). Smoothing will bump up the score of
            terms with low frequency made of many words
        use_frequencies: Whether to use frequencies in nesting calculations (default: False)

    Returns:
        Dictionary that maps lemmatized terms to their ComboBasic scores
        Dictionary that maps lemmatized terms to their occurences in the document
    """
    # Extract terms and frequencies
    term_freqs, occurences = extract_terms(
        doc, n_min=n_min, n_max=n_max, stopwords=stopwords
    )

    # Calculate nesting metrics
    superset_counts, subset_counts = calculate_nesting(
        term_freqs, use_frequencies=use_frequencies
    )

    table_data = []
    headers = [
        "Term",
        "Length",
        "Freq",
        "Supersets",
        "Subsets",
        "Length Score",
        "Nesting Score",
        "Nested Score",
        "Final Score",
    ]

    scores = {}
    for term, freq in term_freqs.items():
        # Length of term in words
        term_length = len(term.split())

        # Calculate score components
        length_score = term_length * math.log(freq + smoothing) if freq > 0 else 0
        nesting_score = alpha * superset_counts[term]  # e_t
        nested_score = beta * subset_counts[term]  # e'_t

        # Combine scores
        scores[term] = length_score + nesting_score + nested_score

        # Add row to table data
        table_data.append(
            [
                term,
                term_length,
                freq,
                superset_counts[term],
                subset_counts[term],
                f"{length_score:.3f}",
                f"{nesting_score:.3f}",
                f"{nested_score:.3f}",
                f"{scores[term]:.3f}",
            ]
        )

    # Sort by final score in descending order
    table_data.sort(key=lambda x: float(x[-1]), reverse=True)

    # Print formatted table
    # print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="grid"))
    with open(
        "/tmp/combobasic.tsv" if beta else "/tmp/basic.tsv", "w", encoding="utf-8"
    ) as f:
        f.write(doc.text + "\n")
        f.write(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="tsv"))

    return scores, occurences


def basic(
    doc: Doc,
    alpha: float = 0.72,  # Original paper suggests 3.5
    n_min: int = 2,
    n_max: int = 6,
    stopwords: Optional[Set[str]] = None,
    smoothing: float = 0,
    use_frequencies: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """
    Implementation of the Basic algorithm for term extraction.

    Basic(t) = |t| * log(f(t)) + α*e_t
    where:
        |t| is the length of term (number of words)
        f(t) is the frequency of the term
        e_t is the number of terms that contain t
        α is a weight parameter

    Args:
        doc: spaCy Doc object
        alpha: Weight for e_t (default: 0.72)
        n_min: Minimum length of terms in tokens (default: 2)
        n_max: Maximum length of terms in tokens (default: 6)
        stopwords: Optional set of stopwords to ignore (default: None)
        smoothing: Smoothing factor for log(f(t)) (default: 0). Smoothing will bump up the score of
            terms with low frequency made of many words
        use_frequencies: Whether to use frequencies in nesting calculations (default: False)
    Returns:
        Dictionary that maps lemmatized terms to their basic scores
        Dictionary that maps lemmatized terms to their occurences in the document
    """

    return combo_basic(
        doc,
        alpha=alpha,
        beta=0,
        n_min=n_min,
        n_max=n_max,
        stopwords=stopwords,
        smoothing=smoothing,
        use_frequencies=use_frequencies,
    )


def cvalue(
    doc: Doc,
    n_min: int = 2,
    n_max: int = 6,
    stopwords: Optional[Set[str]] = None,
    smoothing: float = 0.1,
    use_frequencies: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """Calculate C-Value score for term candidates.

    C-Value promotes term candidates that occur frequently but not as parts of other terms.

    Formula:
    C-Value(t) = log2(|t| + 0.1) * TF(t)                     if t is not nested
    C-Value(t) = log2(|t| + 0.1) * (TF(t) - sum(TF(s))/|S|)  if t is nested

    To use it to match single word terms, set n_min=1 and adjust smoothing to 1.0
    (as per Ventura et al. "Combining c-value and keyword extraction methods for
    biomedical terms extraction.")

    where:
        |t| is the length of term in words
        TF(t) is the term frequency
        s is a term that contains t as a substring
        S is the set of all terms containing t
        |S| is the number of such terms

    Args:
        doc: spaCy Doc object
        n_min: Minimum length of terms in tokens (default: 2)
        n_max: Maximum length of terms in tokens (default: 6)
        stopwords: Optional set of stopwords to ignore (default: None)
        smoothing: Smoothing factor for log2(|t|) (default: 0.1)
        use_frequencies: Whether to use frequencies in nesting calculations (default: False)

    Returns:
        Dictionary that maps lemmatized terms to their cvalue scores
        Dictionary that maps lemmatized terms to their occurences in the document
    """
    # Extract terms and frequencies
    term_freqs, occurences = extract_terms(
        doc, n_min=n_min, n_max=n_max, stopwords=stopwords
    )
    # print(term_freqs, occurences)

    # Calculate nesting metrics
    superset_counts, _ = calculate_nesting(term_freqs, use_frequencies=use_frequencies)

    table_data = []
    headers = [
        "Term",
        "Length",
        "Freq",
        "Supersets",
        "Length Score",
        "Base Frequency Score",
        "Negative Nesting Score",
        "Final Score",
    ]

    scores = {}

    for term, freq in term_freqs.items():
        # Length factor (using log2 as specified in the formula)
        # Added small constant smoothing to handle single-word terms as per Ventura
        avg_container_freq = 0
        term_length = len(term.split())
        length_factor = math.log2(term_length + smoothing)

        # Calculate frequency component
        if superset_counts[term] == 0:
            # Term is not nested in other terms
            frequency_factor = freq
        else:
            # Term is nested in other terms
            # Find all terms that contain this term
            containing_freqs = []
            for container_term, container_freq in term_freqs.items():
                if term != container_term:  # Avoid self
                    # Check if this term appears in the container
                    subterms = set(subphrases(container_term))
                    if term in subterms:
                        containing_freqs.append(container_freq)

            # Calculate average frequency of containing terms
            avg_container_freq = sum(containing_freqs) / len(containing_freqs)
            frequency_factor = freq - avg_container_freq

        scores[term] = length_factor * frequency_factor

        table_data.append(
            [
                term,
                term_length,
                freq,
                superset_counts[term],
                f"{length_factor:.3f}",
                f"{freq:.3f}",
                f"{avg_container_freq:.3f}",
                f"{scores[term]:.3f}",
            ]
        )

    # Sort by final score in descending order
    table_data.sort(key=lambda x: float(x[-1]), reverse=True)

    # Print formatted table
    # print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="grid"))
    with open("/tmp/cvalue.tsv", "w", encoding="utf-8") as f:
        f.write(doc.text + "\n")
        f.write(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="tsv"))

    return scores, occurences


if __name__ == "__main__":  # pragma: no cover
    # Example usage
    # nlp = spacy.load("uk_core_news_trf", disable=["parser", "entity"])
    nlp = spacy.load("de_dep_news_trf", disable=["parser", "entity"])
    text = "This deep neural network uses artificial neural network architecture for deep learning"
    #     text = """Central to the development of cancer are genetic changes that endow these “cancer cells” with many of the
    # hallmarks of cancer, such as self-sufficient growth and resistance to anti-growth and pro-death signals. However, while the
    # genetic changes that occur within cancer cells themselves, such as activated oncogenes or dysfunctional tumor suppressors,
    # are responsible for many aspects of cancer development, they are not sufficient. Tumor promotion and progression are
    # dependent on ancillary processes provided by cells of the tumor environment but that are not necessarily cancerous
    # themselves. Inflammation has long been associated with the development of cancer. This review will discuss the reflexive
    # relationship between cancer and inflammation with particular focus on how considering the role of inflammation in physiologic
    # processes such as the maintenance of tissue homeostasis and repair may provide a logical framework for understanding the U
    # connection between the inflammatory response and cancer."""

    text = "this deep neural network and that deep neural network and shallow neural network"
    text = "Artificial intelligence is a field of science concerned with building computers and machines that can reason, learn, and act in such a way that would normally require human intelligence or that involves data whose scale exceeds what humans can analyze."
    text = """ОНТОЛОГІЇ ЧАСУ: ОГЛЯД І ТЕНДЕНЦІЇ
Час як феномен перебуває у фокусі наукової думки з давніх часів. Він продовжує залишатися важливим предметом дослідження в багатьох дисциплінах через його важливість як базового аспекту для розуміння та формального подання змін. Мета цього аналітичного огляду - з'ясувати, чи формальні подання часу, розроблені на сьогоднішній день, є достатніми щодо вимог фундаментальних і прикладних досліджень у галузі комп'ютерних наук, зокрема, в рамках спільнот штучного інтелекту та семантичного вебу. Щоб проаналізувати, чи існуючі базові теорії, моделі та реалізовані онтології часу добре покривають ці вимоги, було виокремлено та належним чином структуровано набір ознак часу, використовуючи в якості корпусу документів колекцію документів серії симпозіумів TIME. Цей
 набір ознак також допоміг структурувати порівняльний огляд та аналіз найвизначніших теорій часу. В результаті було здійснено відбір підмножини ознак часу (вимог до синтетичної теорії), що відображає 
сентимент спільноти TIME.  Далі було проаналізовано наявні на сьогоднішній день часові логіки, мови подання та онтології з точки зору їхньої зручності використання та охоплення вибраних часових характеристик. Результати показують, що 
оглянуті онтології часу, взяті разом, не задовільно охоплюють деякі важливі характеристики: (i) щільність; (ii) релаксовану лінійність; (iii) фактори масштабу; (iv) точні та періодичні субінтервали; (v) часові міри та годинники.  Зроблено висновок, що необхідні міждисциплінарні зусилля для розгляду особливостей, не охоплених існуючими онтологіями часу, а також щоб гармонізувати подання, що розглядаються по-різному.   
Ключові слова: Час; сентимент; темпоральна характеристика; охоплення; онтологія; подання; вивід.
1.	Вступ
Відомо, що «коли Бог створив час, він створив його багато». Прикметно, що коли йдеться про формальне ставлення до часу, стан справ дуже схожий на цю ірландську приказку.  Час, як явище, був у центрі уваги наукової думки з давніх часів. Сьогодні він продовжує залишатися важливим предметом дослідження для філософів, фізиків, математиків, логіків, комп'ютерників і навіть біологів. Однією з причин, можливо, є те, що час є фундаментальним аспектом для розуміння і реагування на зміни у світі, включаючи найширше різноманіття застосувань, які впливають на еволюцію людства. Отже, прогрес у розумінні світу в його динаміці: (а) ґрунтується на наявності адекватно насиченої і глибокої моделі часу; і (б) підштовхує до подальшого вдосконалення наших моделей часу.  Наприклад, у комп'ютерних науках розвиток штучного інтелекту, баз даних, розподілених систем тощо за останні два десятиліття викликав до життя кілька видатних теоретичних концепцій, що стосуються часових аспектів. Деякі частини цих теорій дали поштовх дослідженням в логіці, що призвело до появи 
сімейства темпоральних логік, які включають в себе 
дескриптивні логіки часу. На основі цього логічного фундаменту мови подання знань отримали можливість репрезентувати час, а 
спільнота Semantic Web реалізувала кілька онтологій часу.  Однак важливо з'ясувати, чи достатньо цього багатства, щоб задовольнити вимоги в дослідженнях і розробках в галузі комп'ютерних наук."""

    text = """Ontologien der Zeit: Übersicht und Trends
Die Zeit als Phänomen steht seit der Antike im Mittelpunkt des wissenschaftlichen Denkens. Sie ist nach wie vor ein wichtiger Forschungsgegenstand in vielen Disziplinen, da sie ein grundlegender Aspekt für das Verständnis und die formale Darstellung von Veränderungen ist. Das Ziel dieses analytischen Überblicks ist es, herauszufinden, ob die bisher entwickelten formalen Darstellungen der Zeit den Bedürfnissen der Grundlagen- und angewandten Forschung in der Informatik und insbesondere innerhalb der Artificial Intelligence und Semantic Web Communities genügen. Um zu analysieren, ob die existierenden grundlegenden Theorien, Modelle und implementierten Ontologien der Zeit diese Bedürfnisse gut abdecken, wurde die Menge der Merkmale der Zeit extrahiert und angemessen strukturiert, indem die Papiersammlung der TIME Symposienreihe als Dokumentenkorpus verwendet wurde. Dieses Merkmalsset half weiter, die vergleichende Überprüfung und Analyse der prominentesten Zeittheorien zu strukturieren. Infolgedessen wurde die Auswahl der Teilmenge der Zeitmerkmale (die Anforderungen an eine Synthetische Theorie) unter Berücksichtigung der Meinung der TIME-Gemeinschaft getroffen.  Des Weiteren wurden die bisher verfügbaren temporalen Logiken, Repräsentationssprachen und Ontologien hinsichtlich ihrer Benutzerfreundlichkeit und der Abdeckung der ausgewählten temporalen Merkmale untersucht. Die Ergebnisse zeigen, dass die untersuchten Ontologien der Zeit zusammengenommen einige wichtige Merkmale nicht zufriedenstellend abdecken: (i) Dichte; (ii) entspannte Linearität; (iii) Skalenfaktoren; (iv) ordentliche und periodische Subintervalle ; (v) temporale Maße und Uhren.  Man ist zu dem Schluss gekommen, dass eine disziplinübergreifende Anstrengung erforderlich ist, um die von den bestehenden Ontologien der Zeit nicht abgedeckten Merkmale zu behandeln und auch die unterschiedlich behandelten Darstellungen zu harmonisieren.   
Schlüsselwörter: Zeit; Stimmung; zeitliches Merkmal; Abdeckung; Ontologie; Darstellung; Schlussfolgerungen.

Einführung
Es ist bekannt, dass "als Gott die Zeit schuf, schuf er reichlich von ihr". Bemerkenswerterweise folgt der Status, wenn es um die formale Behandlung der Zeit geht, sehr stark diesem irischen Sprichwort.  Die Zeit als Phänomen steht seit der Antike im Mittelpunkt des wissenschaftlichen Denkens. Auch heute noch ist sie ein wichtiger Forschungsgegenstand für Philosophen, Physiker, Mathematiker, Logiker, Informatiker und sogar Biologen. Ein Grund dafür ist vielleicht, dass die Zeit ein grundlegender Aspekt ist, um Veränderungen in der Welt zu verstehen und darauf zu reagieren, einschließlich der unterschiedlichsten Anwendungen, die sich auf die Entwicklung der Menschheit auswirken. Der Fortschritt beim Verständnis der Welt in ihrer Dynamik basiert also (a) auf einem ausreichend reichhaltigen und tiefen Modell der Zeit; und (b) treibt die weitere Verfeinerung unserer Zeitmodelle voran.  In der Informatik zum Beispiel haben die Entwicklungen in den Bereichen künstliche Intelligenz, Datenbanken, verteilte Systeme usw. in den letzten zwei Jahrzehnten mehrere herausragende theoretische Rahmenwerke hervorgebracht, die sich mit zeitlichen Aspekten befassen. Einige Teile dieser Theorien gaben der Forschung im Bereich der Logik Auftrieb und führten zu einer Familie von temporalen Logiken , die temporale Beschreibungslogiken umfasst. Auf dieser logischen Grundlage haben Wissensrepräsentationssprachen ihre Fähigkeit zur Darstellung von Zeit erhalten, und mehrere Zeit-Ontologien wurden von der Semantic Web-Gemeinschaft implementiert.  Es ist jedoch wichtig, herauszufinden, ob diese Fülle ausreicht, um den Anforderungen der Informatikforschung und -entwicklung gerecht zu werden."""

    test_scores, _ = cvalue(nlp(text), use_frequencies=False)

    print("C-Value scores:")
    # Print scores in descending order
    for a_term, a_score in sorted(
        test_scores.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{a_term}: {a_score:.3f}")

    # test_scores, _ = combo_basic(nlp(text), use_frequencies=False)

    # print("ComboBasic scores:")
    # # Print scores in descending order
    # for a_term, a_score in sorted(test_scores.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{a_term}: {a_score:.3f}")

    # test_scores, _ = basic(nlp(text), use_frequencies=False)

    # print("Basic scores:")
    # # Print scores in descending order
    # for a_term, a_score in sorted(test_scores.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{a_term}: {a_score:.3f}")
