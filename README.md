# PA-X Peace Agreement Corpus Search

A Streamlit web application for searching and analysing the full [PA-X Peace Agreement Database](https://www.peaceagreements.org/) corpus of ~2,257 peace agreements.

Built by [PeaceRep, University of Edinburgh](https://peacerep.org/).

---

## Features

- **Keyword & phrase search** across full agreement texts, with comma-separated multi-term input
- **Boolean search modes** — OR (default), AND (terms co-occur in the same sentence), NOT (first term present, remaining absent)
- **Exact phrase matching** for multi-word expressions
- **Word variant matching** — approximate stem-based lemmatisation for single-word terms (e.g. *internal* also matches *internally*)
- **Pre-search filters** — region, country/conflict, agreement stage, agreement type, year range, and a three-level topic/issue/sub-issue hierarchy
- **Topic-scoped search** — optionally restrict search to text segments already tagged for the selected topics
- **Sentence-level results** — matched snippets with in-text highlighting, per-term counts, and agreement-level summary metrics
- **Mention review editor** — flag individual sentence-level mentions for exclusion and add notes before export
- **CSV export** — download all mentions or only reviewed/kept mentions
- **Visualisations** — charts by year, region, stage, and country, filterable by individual search term
- **Per-agreement detail** — expandable cards with direct links to PA-X and agreement PDFs

---

## Repository structure

```
corpus_app.py                  # Main Streamlit application
requirements.txt               # Python dependencies
assets/
  logos/                       # PeaceRep and PA-X branding images
pax/
  pax.csv                      # Agreement metadata (region, country, stage, year, links)
  pax_corpus_2257_agreements_v10.csv   # Full text corpus
  only_tagged_pax_topic_codes_segments.csv  # Topic-tagged text segments
archive/                       # Superseded versions (local only, gitignored)
```

---

## Local development

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/) or [uv](https://github.com/astral-sh/uv)

### Setup

```bash
git clone https://github.com/peacerep/corpus_app.git
cd corpus_app

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run corpus_app.py
```

The app will open at `http://localhost:8501`.

---

## Data files

| File | Description |
|---|---|
| `pax/pax.csv` | Agreement-level metadata: title, year, region, country, stage, agreement type, PA-X and PDF hyperlinks |
| `pax/pax_corpus_2257_agreements_v10.csv` | Full agreement texts keyed by `AgtId` |
| `only_tagged_pax_topic_codes_segments.csv` | Text segments tagged with PA-X topic codes, used for topic filtering and topic-scoped search |

Data source: Bell, C., & Badanjak, S. (2019). Introducing PA-X: A new peace agreement database and dataset. *Journal of Peace Research*, 56(3), 452–466.

---

## Deployment on Streamlit Cloud

### Initial deploy

1. Push the repository to GitHub (e.g. `github.com/peacerep/corpus_app`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app** and configure:
   - **Repository:** `peacerep/corpus_app`
   - **Branch:** `main`
   - **Main file path:** `corpus_app.py`
4. Click **Deploy**.

### Updating a live deployment

Streamlit Cloud automatically redeploys when changes are pushed to the configured branch:

```bash
git add .
git commit -m "your message"
git push origin main
```

The app will rebuild and restart within a minute or two.

### Forcing a clean rebuild

If the app is showing a stale version or dependency errors after a push:

1. Go to the app in the [Streamlit Cloud dashboard](https://share.streamlit.io).
2. Click the **⋮** menu (top-right of the app tile) → **Settings**.
3. Under **General**, confirm the **Main file path** is set to `corpus_app.py`.
4. Return to the dashboard, click **⋮** → **Delete cache and reboot**.

A plain **Reboot** restarts the process but reuses the cached package install. **Delete cache and reboot** forces a full fresh install from `requirements.txt`.

### Secrets

This app does not currently use any API keys or secrets. If secrets are needed in future, add them via **Settings → Secrets** in the Streamlit Cloud dashboard and access them with `st.secrets`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `pandas` | Data loading and manipulation |
| `plotly` | Interactive charts |
| `nltk` | Snowball stemmer for word-variant matching |
| `spacy` | NLP pipeline (en_core_web_sm model) |

See [requirements.txt](requirements.txt) for pinned versions.

---

## Licence

Data: [PA-X Peace Agreement Database](https://www.peaceagreements.org/) — University of Edinburgh.  
Code: Niamh Henry, PeaceRep, University of Edinburgh.
