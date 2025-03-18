from keybert import KeyBERT

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

doc = """Ontologies of Time: Review and Trends
    Time, as a phenomenon, has been in the focus of scientific thought from ancient times. It continues to be an important subject of research in many disciplines due to its importance as a basic aspect for understanding and formally representing change. The goal of this analytical review is to find out if the formal representations of time developed to date suffice to the needs of the basic and applied research in Computer Science, and in particular within the Artificial Intelligence and Semantic Web communities. To analyze if the existing basic theories, models, and implemented ontologies of time cover these needs well, the set of the features of time has been extracted and appropriately structured using the paper collection of the TIME Symposia series as the document corpus. This feature set further helped to structure the comparative review and analysis of the most prominent temporal theories. As a result, the selection of the subset of the features of time (the requirements for a Synthetic Theory) has been made reflecting the TIME community sentiment.  Further, the temporal logics, representation languages, and ontologies available to date, have been reviewed regarding their usability aspects and the coverage of the selected temporal features. The results reveal that the reviewed ontologies of time taken together do not satisfactorily cover some important features: (i) density; (ii) relaxed linearity; (iii) scale factors; (iv) proper and periodic subintervals; (v) temporal measures and clocks.  It has been concluded that a cross-disciplinary effort is required to address the features not covered by the existing ontologies of time, and also harmonize the representations addressed differently.   
    Keywords: Time; sentiment; temporal feature; coverage; ontology; representation; reasoning.
    Introduction
    It is acknowledged that “when God made time, he made plenty of it”. Remarkably, when it goes about the formal treatment of time, the status is very much following this Irish saying.  Time, as a phenomenon, has been in the focus of scientific thought from ancient times. Today it continues to be an important subject of research for philosophers, physicists, mathematicians, logicians, computer scientists, and even biologists. One reason, perhaps, is that time is a fundamental aspect to understand and react to change in the World, including the broadest diversity of applications that impact the evolution of the Humankind. So, the progress in understanding the World in its dynamics: (a) is based on having an adequately rich and deep model of time; and (b) pushes forward the further refinement of our time models.  For example, in Computer Science the developments in Artificial Intelligence, Databases, Distributed Systems, etc. in the last two decades have brought to life several prominent theoretical frameworks dealing with temporal aspects. Some parts of these theories gave boost to the research in logics – yielding a family of temporal logics, comprising temporal description logics. Based on this logical foundation, knowledge representation languages have received their capability to represent time, and several ontologies of time have been implemented by the Semantic Web community.  It is however important to find out if this plenty is enough to meet the requirement in Computer Science research and development.
    The objective of this analytic review paper is to answer this question – i.e. to find out if the formal representations of time developed to date suffice to the needs of coping with different aspects of change. The remainder of the paper is structured as follows. 
    """
res = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(1, 6),
    top_n=50,
    stop_words="english",
    diversity=0.1,
)

print(res)
