# AI Concepts Used in This Project

## What is Retrieval-Augmented Generation (RAG)?

A RAG system answers questions by first *retrieving* relevant passages
from a knowledge base (using similarity search over document embeddings or
term statistics) and then *generating* an answer grounded in those
retrieved passages, rather than relying purely on a language model's
training data. This reduces hallucination and lets the system cite its
sources. This project's chatbot is a RAG system: it searches a small,
curated knowledge base (this file, plus the NISQ/IBM Runtime/technique
articles alongside it, plus the project's own README and REPORT) using
TF-IDF (term frequency-inverse document frequency) similarity search, then
either passes the retrieved passages to a language model to compose a
grounded answer, or -- if no LLM API key is configured -- returns the
retrieved passages directly as an extractive answer.

## What is a regression model?

A regression model learns a function that maps input features to a
continuous numeric output, by fitting to historical examples where both
the inputs and the true output are known. This project trains a
**Random Forest Regressor** (an ensemble of decision trees, each trained
on a random subset of data/features, whose predictions are averaged) to
predict how much *absolute error* a given mitigation technique is likely
to leave behind, as a function of noise strength, qubit count, and circuit
type. This is used to rank the six implemented techniques and recommend
the one predicted to perform best under given conditions.

## What is Clifford Data Regression, specifically?

CDR (one of the six mitigation techniques above) is itself a form of
regression: it fits a model -- classically, often just a linear
regression -- mapping *noisy* expectation values (from near-Clifford
circuits, which are cheap to simulate exactly) to their *known-exact*
values, then applies that fitted map to the noisy result of the real
circuit being studied. It's a nice example of "regression" showing up at
two different layers of this project: once inside a mitigation technique
itself (CDR), and again in the separate technique-recommendation model
described above.

## What is a Large Language Model (LLM), and how is one used here?

An LLM is a neural network (typically a Transformer) trained on very large
text corpora to predict and generate natural language. This project's
chatbot and "explain this result" feature can optionally call a hosted LLM
(Anthropic's Claude, via API) to turn retrieved technical passages and raw
numeric results into a plain-English explanation. This is optional and
pluggable: if no API key is configured in the environment, the project
falls back to template-based explanations and direct retrieval results,
so the features remain functional (if less fluent) without any external
API dependency or cost.

---
*General AI/ML concepts summarized from standard machine learning and NLP
literature; scikit-learn documentation for Random Forest implementation
details.*
