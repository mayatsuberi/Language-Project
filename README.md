# Cross-Lingual Semantic Representations in the Brain

This project asks whether the brain encodes **language-independent (amodal)
semantic content** during naturalistic speech comprehension, or whether
apparent "cross-lingual" neural signal is better explained by a simpler
methodological confound (model capacity / design width).

We use intracranial ECoG recordings from subjects listening to an English
podcast, together with English, Hebrew, and Arabic contextual embeddings of
the same words (Hebrew/Arabic obtained via translation). For each foreign
language we compute the **residual** — the part of the Hebrew/Arabic
embedding space that cannot be linearly predicted from English — and test
whether that residual predicts neural activity above a dimensionality-matched
control. If cross-lingual semantics were truly reflected in the brain signal,
the residual should out-predict the control; the pipeline is built to make
that comparison rigorously (permutation tests, FDR correction, and a
width-matched shift control) rather than assume it.

## Data

- **Neural data:** [`ds005574`](https://openneuro.org/datasets/ds005574) — the
  "Podcast" ECoG dataset (Zada, Nastase, Aubrey, et al., Hasson Lab, NYU),
  openly available on OpenNeuro (CC0). Subjects listened to a ~30-minute
  English podcast while ECoG was recorded. See
  [hassonlab.github.io/podcast-ecog-tutorials](https://hassonlab.github.io/podcast-ecog-tutorials)
  and the [podcast-ecog-paper repo](https://github.com/hassonlab/podcast-ecog-paper).
- **Stimulus text:** the English podcast transcript (1,735 content words
  across 402 sentences), professionally translated into Hebrew and Arabic
  and time-aligned to the original audio (`data/sentences/`).
- **Static baseline embeddings:** FastText vectors (`data/processed/podcast_trilingual_embeddings.csv`),
  trained from scratch on large in-language corpora as part of a related
  Amirim project, used as a static (non-contextual) comparison condition.

Raw/derivative ECoG data and large embedding/model files are not tracked in
git (see `.gitignore`); `data/ds005574` must be downloaded separately (e.g.
via [`openneuro-py`](https://github.com/hasso-lab/podcast-ecog-tutorials) or
the OpenNeuro CLI/DataLad) and placed at that path.

## Method overview

1. **Contextual embeddings** are extracted with multilingual/multilingual-capable
   transformer models under several context regimes: sentence context, a
   ±8-word sliding window, and full-transcript causal context (XGLM-1.7B,
   GemmaX2-28-2B). Hebrew/Arabic word positions inside a sentence are located
   via cross-lingual cosine similarity against XLM-RoBERTa representations of
   the isolated word.
2. **Projection & residualization** (notebook 05): for each language and
   embedding mode, fit `English @ W ≈ Foreign` by least squares, then take
   `residual = Foreign - English @ W`. The residual is, by construction,
   orthogonal to the English space and is the cross-lingual signal under test.
3. **Encoding models** (notebook 06): ridge/banded regression (via
   [`himalaya`](https://github.com/gallantlab/himalaya)) predicts ECoG activity
   from feature blocks (English PCA components, residual PCA components,
   etc.), evaluated per subject/electrode/time-lag with cross-validation and
   a circular-shift permutation null.
4. **Statistics & figures** (notebook 07): FDR-corrected paired t-tests
   compare each residual condition against (a) English alone and (b) a
   dimensionality-matched shift control, which isolates genuine cross-lingual
   information from the capacity cost of a wider design matrix.

## Repository structure

```
notebooks/
  01_Generate_Contextualized_Embeddings_EN.ipynb   English contextual embeddings (XLM-RoBERTa, sentence context)
  02_Generate_Embeddings_HE.ipynb                  Hebrew contextual embeddings
  03_Generate_Embeddings_AR.ipynb                  Arabic contextual embeddings
  04_Generate_Sliding_Window_Embeddings.ipynb      ±8-word sliding-window embeddings, all 3 languages
  04b_Generate_XGLM_Embeddings.ipynb               Full-transcript causal embeddings (XGLM-1.7B)
  04c_Generate_GemmaX2_Embeddings.ipynb            Full-transcript causal embeddings (GemmaX2-28-2B)
  05_Projection_Residuals_Contextual.ipynb         English→Foreign projection & residual computation
  06_Encoding_Contextual.ipynb                     Ridge encoding models + permutation tests
  07_Results_Visualization.ipynb                   Statistics, figures, and summary tables

src/
  static_encoding.py    Core encoding pipeline (process_embeddings, circular-shift null) — used by notebook 06
  block_pipeline.py     Fold-safe per-block PCA/residual feature transformer — used by static_encoding.py
  residual_spectrum.py  Standalone diagnostic: residual PCA spectra across embedding modes

data/
  ds005574/            OpenNeuro "Podcast" ECoG dataset (download separately, see above)
  sentences/            Aligned EN/HE/AR transcript sentences, word-to-sentence maps, and the
                        1,735-word filtered transcript (translated_podcast_transcript_filtered.csv)
  processed/            Generated embeddings, projections, residuals, manifests (pipeline output);
                        also the tracked FastText baseline embeddings CSV
  Amirim_Project_Submission/   Legacy FastText training data/models from a prior project — not
                        needed to run the pipeline, kept locally only, git-ignored

results/
  encoding_*/           Per-condition encoding outputs (correlations, permutation nulls) per subject
  figures*/              Final figures (PNG/PDF) for the report
  table*.csv, residual_*.csv/json   Summary statistics feeding notebook 07
```

## Setup

Using conda (recommended, matches `environment.yml`):

```bash
conda env create -f environment.yml
conda activate language_project_env
```

Or with pip (`requirements.txt`):

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `mne` / `mne-bids` / `pybids` (ECoG/BIDS I/O), `torch` +
`transformers` / `huggingface_hub` (contextual embedding models),
`fasttext-wheel` (static embeddings), `scikit-learn` / `statsmodels` /
`himalaya` (encoding models & statistics), `jupyterlab`.

## Running the pipeline

Notebooks are numbered in execution order. `01`–`04c` generate embeddings
(GPU recommended for `04b`/`04c`), `05` computes projections/residuals across
all available modes, `06` runs the encoding models per subject (writes to
`results/`), and `07` produces the final figures and statistical tables.
Each notebook skips already-completed work where possible, so re-running
after an interruption is safe. Notebook 06 tags its output directories with a
`RUN_TAG`, so different pipeline configurations never silently overwrite each
other's results.

## Citation

If you use the ECoG data, please cite the OpenNeuro dataset:

> Zada, Z., Nastase, S. A., Aubrey, B., et al. (2023). The "Podcast" ECoG
> dataset. OpenNeuro. https://doi.org/10.18112/openneuro.ds005574.v1.0.2
