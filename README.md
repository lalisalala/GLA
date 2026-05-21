# GLA Metadata Field Analysis

This repository contains research code and data for evaluating how different metadata fields affect semantic retrieval over Greater London Authority (GLA) / London Datastore-style dataset records.

The project builds FAISS indices from different metadata representations, such as titles, keywords, topics, descriptions, and enriched variants, then evaluates retrieval performance against user queries.

## Project overview

The goal of this project is to compare how well different metadata-field combinations support semantic dataset search.

The repository currently includes:

- `data/` — JSON files containing dataset metadata variants and user query sets.
- `models/faiss_search.py` — builds FAISS indices for each metadata ablation.
- `models/faiss_indices/` — generated FAISS indices and ID mappings.
- `evaluation/evaluate_faiss.py` — evaluates retrieval performance.
- `evaluation/` — generated plots and CSV outputs from the evaluation runs.
- `LICENSE` — project license.

## Installation

```bash
git clone https://github.com/lalisalala/GLA.git
cd GLA
code .
python -m venv .venv