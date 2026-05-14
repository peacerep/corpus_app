import streamlit as st
import pandas as pd
import re
import base64
import html as html_module
import plotly.express as px
import time
from nltk.stem import SnowballStemmer

@st.cache_data
def load_topic_segments():
    df = pd.read_csv("pax/only_tagged_pax_topic_codes_segments.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["max"] = pd.to_numeric(df["max"], errors="coerce").fillna(0)
    for column in ["category", "issue_label", "subissue_label", "type", "tagged_text"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    return df
st.set_page_config(page_title="PA-X Corpus Search", layout="wide")

STEMMER = SnowballStemmer("english")
WORD_RE = re.compile(r"\b[\w'-]+\b")


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_local_image(path):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        suffix = path.split(".")[-1].lower()
        mime = "image/png" if suffix == "png" else "image/jpeg"
        return f"data:{mime};base64,{data}"
    except FileNotFoundError:
        return ""


def render_link_button(label, url, secondary=False):
    if not url or (isinstance(url, float) and pd.isna(url)):
        return ""
    cls = "corpus-link-button secondary" if secondary else "corpus-link-button"
    safe_label = html_module.escape(str(label))
    safe_url = html_module.escape(str(url), quote=True)
    return (
        f'<a class="{cls}" href="{safe_url}" target="_blank" '
        f'rel="noopener noreferrer">{safe_label}</a>'
    )


def format_topic_level_label(score):
    return {1: "Any mention", 2: "Rhetorical or stronger", 3: "Substantive only"}.get(
        score, "Any mention"
    )


def count_label(value, singular, plural=None):
    plural = plural or f"{singular}s"
    return f"{value:,} {singular if value == 1 else plural}"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_pax():
    df = pd.read_csv("pax/pax.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


@st.cache_data
def load_corpus():
    df = pd.read_csv("pax/pax_corpus_2257_agreements_v10.csv")
    return df


@st.cache_data
def load_topics():
    """Derive unique topic rows from the segments file (max value per agreement×topic group)."""
    seg = load_topic_segments()
    group_cols = ["AgtId", "category", "issue_label", "subissue_label", "type"]
    df = (
        seg.groupby(group_cols, as_index=False)["value"]
        .max()
    )
    # Restore max column (max possible value for the type) — take first non-zero value per type
    max_vals = seg.groupby("type", as_index=False)["max"].max().rename(columns={"max": "max"})
    df = df.merge(max_vals, on="type", how="left")
    return df


# ── Sentence splitting ────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def is_single_word_term(term):
    return bool(term) and len(term.split()) == 1


def stem_word(value):
    return STEMMER.stem(value.lower())


def term_matches_text(term, text, use_lemmas=False):
    if not use_lemmas or not is_single_word_term(term):
        return re.search(re.escape(term), text, re.IGNORECASE) is not None

    term_stem = stem_word(term)
    return any(stem_word(match.group()) == term_stem for match in WORD_RE.finditer(text))


def count_term_matches(term, text, use_lemmas=False):
    if not use_lemmas or not is_single_word_term(term):
        return len(re.findall(re.escape(term), text, re.IGNORECASE))

    term_stem = stem_word(term)
    return sum(1 for match in WORD_RE.finditer(text) if stem_word(match.group()) == term_stem)


# ── Highlighting ──────────────────────────────────────────────────────────────

def highlight_terms(sentence, terms, use_lemmas=False):
    if use_lemmas:
        single_word_stems = {
            stem_word(term)
            for term in terms
            if is_single_word_term(term)
        }
        parts = []
        last_end = 0
        for match in WORD_RE.finditer(sentence):
            parts.append(html_module.escape(sentence[last_end:match.start()]))
            token = match.group()
            token_html = html_module.escape(token)
            if stem_word(token) in single_word_stems:
                token_html = (
                    '<mark style="background:#ffd966;padding:0 2px;border-radius:2px;">'
                    + token_html
                    + "</mark>"
                )
            parts.append(token_html)
            last_end = match.end()
        parts.append(html_module.escape(sentence[last_end:]))
        result = "".join(parts)
    else:
        result = html_module.escape(sentence)

    for term in terms:
        if use_lemmas and is_single_word_term(term):
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(
            lambda m: (
                '<mark style="background:#ffd966;padding:0 2px;border-radius:2px;">'
                + html_module.escape(m.group())
                + "</mark>"
            ),
            result,
        )
    return result


# ── Search ────────────────────────────────────────────────────────────────────

def search_corpus(corpus_df, pax_df, term_list, use_boolean, boolean_mode, progress_bar, text_column="Agreement text", use_lemmas=False):
    """Search corpus_df for term_list. Returns agreement results, snippets, and mention rows."""
    escaped = [re.escape(t) for t in term_list]
    per_term = [re.compile(e, re.IGNORECASE) for e in escaped]

    if use_boolean and boolean_mode == "AND":
        def agreement_check(text):
            return all(term_matches_text(term, text, use_lemmas=use_lemmas) for term in term_list)
        def sent_check(s):
            return all(term_matches_text(term, s, use_lemmas=use_lemmas) for term in term_list)
    elif use_boolean and boolean_mode == "NOT" and per_term:
        def agreement_check(text):
            return term_matches_text(term_list[0], text, use_lemmas=use_lemmas)
        def sent_check(s):
            return term_matches_text(term_list[0], s, use_lemmas=use_lemmas) and all(
                not term_matches_text(term, s, use_lemmas=use_lemmas) for term in term_list[1:]
            )
    else:  # OR (default)
        combined = re.compile("|".join(escaped), re.IGNORECASE)
        def agreement_check(text):
            return any(term_matches_text(term, text, use_lemmas=use_lemmas) for term in term_list) if use_lemmas else combined.search(text)
        def sent_check(s):
            return any(term_matches_text(term, s, use_lemmas=use_lemmas) for term in term_list) if use_lemmas else combined.search(s)

    results_map = {}
    all_snippets = {}
    mention_rows = []
    n = len(corpus_df)
    pax_indexed = pax_df.set_index("AgtId")

    for i, (_, row) in enumerate(corpus_df.iterrows()):
        if n:
            progress_bar.progress((i + 1) / n)
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        if not agreement_check(text):
            continue
        sentences = split_sentences(text)
        matched = [s for s in sentences if sent_check(s)]
        if not matched:
            continue

        term_counts = {
            term: sum(count_term_matches(term, sentence, use_lemmas=use_lemmas) for sentence in matched)
            for term in term_list
        }
        total = sum(term_counts.values())
        agt_id = row["AgtId"]

        if agt_id in pax_indexed.index:
            m = pax_indexed.loc[agt_id]
            title = m.get("Agt", "")
            year = m.get("year", "")
            stage = m.get("stage_label", "")
            region = m.get("Reg", "")
            country = m.get("Con", "")
            pax_link = m.get("PAX_Hyperlink", "")
            pdf_link = m.get("PDF_Hyperlink", "")
        else:
            title = row.get("Agt", str(agt_id))
            year = row.get("year", "")
            stage = region = country = pax_link = pdf_link = ""

        if agt_id not in results_map:
            results_map[agt_id] = {
                "AgtId": agt_id,
                "Agreement": title,
                "Year": year,
                "Stage": stage,
                "Region": region,
                "Country / Conflict": country,
                "Total mentions": 0,
                **{f'"{t}"': 0 for t in term_list},
                "PA-X Link": pax_link,
                "PDF Link": pdf_link,
            }
            all_snippets[agt_id] = []

        results_map[agt_id]["Total mentions"] += total
        for term, count in term_counts.items():
            results_map[agt_id][f'"{term}"'] += count

        all_snippets[agt_id].extend(highlight_terms(s, term_list, use_lemmas=use_lemmas) for s in matched)

        for sentence in matched:
            sentence_term_counts = {
                term: count_term_matches(term, sentence, use_lemmas=use_lemmas)
                for term in term_list
            }
            matched_terms = [
                f"{term} ({count})"
                for term, count in sentence_term_counts.items()
                if count > 0
            ]
            mention_rows.append(
                {
                    "AgtId": agt_id,
                    "Agreement": title,
                    "Year": year,
                    "Stage": stage,
                    "Region": region,
                    "Country / Conflict": country,
                    "Matched sentence": sentence,
                    "Matched terms": ", ".join(matched_terms),
                    "Exclude": False,
                    "Notes": "",
                    "PA-X Link": pax_link,
                    "PDF Link": pdf_link,
                }
            )

    results = sorted(
        results_map.values(),
        key=lambda item: (-item["Total mentions"], str(item["Agreement"])),
    )
    return results, all_snippets, mention_rows


def build_mention_rows_from_results(results, snippets, term_list):
    mention_rows = []
    term_patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in term_list]

    for row in results:
        agt_id = row.get("AgtId")
        for snippet in snippets.get(agt_id, []):
            sentence = re.sub(r"<[^>]+>", "", snippet)
            sentence_term_counts = {
                term: len(pattern.findall(sentence))
                for term, pattern in zip(term_list, term_patterns)
            }
            matched_terms = [
                f"{term} ({count})"
                for term, count in sentence_term_counts.items()
                if count > 0
            ]
            mention_rows.append(
                {
                    "AgtId": agt_id,
                    "Agreement": row.get("Agreement", ""),
                    "Year": row.get("Year", ""),
                    "Stage": row.get("Stage", ""),
                    "Region": row.get("Region", ""),
                    "Country / Conflict": row.get("Country / Conflict", ""),
                    "Matched sentence": sentence,
                    "Matched terms": ", ".join(matched_terms),
                    "Exclude": False,
                    "Notes": "",
                    "PA-X Link": row.get("PA-X Link", ""),
                    "PDF Link": row.get("PDF Link", ""),
                }
            )

    return mention_rows


# ── Header HTML ───────────────────────────────────────────────────────────────

peace_rep_logo = encode_local_image("assets/logos/PeaceRep_white.jpg")
pax_logo = encode_local_image("assets/logos/Pax_white.png")

_logo_left = (
    f'<img src="{peace_rep_logo}" alt="PeaceRep Logo" />' if peace_rep_logo else ""
)
_logo_right = (
    f'<img class="pax-logo" src="{pax_logo}" alt="PA-X Logo" />' if pax_logo else ""
)

HEADER_HTML = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        body { font-family: 'Montserrat', sans-serif; color: #091f40; }
        .block-container { padding-top: 3.25rem !important; }
        .top-banner {
            background: #091f40;
            border-radius: 0 0 20px 20px;
            padding: 1.45rem 2.9rem 1.35rem 2.9rem;
            margin: 0 calc(50% - 50vw) 1.5rem;
            overflow: visible;
        }
        .header-container {
            display: grid;
            grid-template-columns: 280px minmax(0, 1fr) 280px;
            align-items: center;
            column-gap: 2.2rem;
            max-width: 1720px;
            margin: 0 auto;
            padding-top: 0.55rem;
        }
        .header-brand {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            height: 80px;
        }
        .header-logo-group {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            height: 80px;
            justify-self: end;
        }
        .header-brand img {
            width: 220px;
            max-height: 72px;
            object-fit: contain;
            object-position: left center;
            display: block;
        }
        .header-logo-group img.pax-logo {
              width: 220px;
              max-height: 76px;
            object-fit: contain;
            display: block;
        }
        .header-title {
            text-align: center;
            font-size: 2.15em;
            margin: 0;
            font-family: 'Montserrat', sans-serif;
            color: #ffffff !important;
            line-height: 1.18;
            max-width: 900px;
            justify-self: center;
        }
        .header-divider {
            width: min(100%, 1720px);
            max-width: 1720px;
            margin: 1rem auto 1.15rem auto;
            border: none;
            height: 6px;
            background: linear-gradient(
                90deg,
                rgba(255,255,255,0.18) 0%,
                rgba(255,255,255,0.98) 14%,
                rgba(255,255,255,0.98) 86%,
                rgba(255,255,255,0.18) 100%
            );
            box-shadow: 0 0 20px rgba(255,255,255,0.28);
            border-radius: 999px;
            display: block;
        }
        .sub-title {
            text-align: center;
            font-size: 1.05em;
            font-family: 'Montserrat', sans-serif;
            color: #e8eef7;
            max-width: 1380px;
            margin: 0 auto;
        }
        .sub-title p { margin-bottom: 0.45rem; }
        .corpus-link-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.55rem 0.95rem;
            border-radius: 999px;
            border: 1px solid #091f40;
            background: #091f40;
            color: #ffffff !important;
            text-decoration: none;
            font-weight: 700;
            margin-right: 0.5rem;
            margin-top: 0.5rem;
            font-family: 'Montserrat', sans-serif;
            font-size: 0.9em;
        }
        .corpus-link-button.secondary {
            background: #ffffff;
            color: #091f40 !important;
            border-color: #091f40;
        }
        .corpus-link-button:hover { opacity: 0.88; }
        .corpus-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin: 0 0 0.65rem 0;
        }
        .agreement-title {
            margin: 0.2rem 0 0.35rem 0;
            font-size: 1.25rem;
            font-weight: 700;
            color: #091f40;
            line-height: 1.25;
        }
        .snippet-block {
            background: #f7f9fc;
            border-left: 3px solid #091f40;
            padding: 0.5rem 0.75rem;
            margin: 0.3rem 0;
            border-radius: 0 4px 4px 0;
            font-size: 0.92em;
            line-height: 1.5;
        }
        @media (max-width: 900px) {
            .header-container {
                display: flex;
                flex-direction: column;
                text-align: center;
                align-items: center;
                row-gap: 0.85rem;
            }
            .header-brand, .header-logo-group {
                width: 100%;
                height: auto;
                justify-content: center;
            }
            .header-brand img, .header-logo-group img { width: 150px; }
            .header-logo-group img.pax-logo { width: 180px; }
            .header-title { font-size: 1.7em; margin-top: 0; max-width: none; }
            .top-banner { padding: 1.15rem 1rem 1.25rem 1rem; }
        }
    </style>
    <div class="top-banner">
        <div class="header-container">
            <div class="header-brand">__PEACEREP_LOGO__</div>
            <h1 class="header-title">PA-X Peace Agreement Corpus Search</h1>
            <div class="header-logo-group">__PAX_LOGO__</div>
        </div>
        <hr class="header-divider" />
        <div class="sub-title">
            <p><b>Credits: Bell, C., &amp; Badanjak, S. (2019). Introducing PA-X: A new peace agreement
            database and dataset. Journal of Peace Research, 56(3), 452&#8211;466.
                Available at <a href="https://www.peaceagreements.org/" target="_blank" rel="noopener noreferrer" style="color:#ffffff; text-decoration:underline;">https://www.peaceagreements.org/</a> &#8212; PeaceRep, University of Edinburgh</b></p>
            <p>Search for keywords and phrases across the full PA-X corpus of ~2,257 peace agreements.
            Filter by region, country, stage, type, year, and topic before searching.
            Results include matched sentence snippets, per-term counts, and downloadable CSV export.</p>
        </div>
    </div>
""".replace("__PEACEREP_LOGO__", _logo_left).replace("__PAX_LOGO__", _logo_right)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    pax_df = load_pax()
    corpus_df = load_corpus()
    topics_df = load_topics()
    topic_segments_df = load_topic_segments()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("### Filter Agreements")
    st.caption(
        "Use the filters to narrow the corpus before searching. All filters are optional, "
        "and the search will run on the full corpus if none are selected."
    )

    f1l, f1r = st.columns(2)
    f2l, f2r = st.columns(2)
    f3l, f3r = st.columns(2)

    with f1l:
        selected_regions = st.multiselect(
            "Region",
            options=sorted(pax_df["Reg"].dropna().astype(str).unique().tolist()),
            help="PA-X regional groupings.",
        )
    with f1r:
        selected_conflicts = st.multiselect(
            "Country / Conflict",
            options=sorted(pax_df["Con"].dropna().astype(str).unique().tolist()),
            help="Filter by the PA-X conflict/country field.",
        )
    with f2l:
        selected_stages = st.multiselect(
            "Stage",
            options=sorted(pax_df["stage_label"].dropna().astype(str).unique().tolist()),
            help="PA-X stage labels.",
        )
    with f2r:
        selected_agt_types = st.multiselect(
            "Agreement Type",
            options=sorted(pax_df["agt_type"].dropna().astype(str).unique().tolist()),
            help="Readable agreement type labels from PA-X.",
        )

    year_values = sorted(pax_df["year"].dropna().astype(int).unique().tolist())
    with f3l:
        year_range = (
            st.select_slider(
                "Year range",
                options=year_values,
                value=(year_values[0], year_values[-1]),
                help="Filter agreements by year range.",
            )
            if year_values
            else None
        )

    with f3r:
        # Placeholder label — the actual topic expander is full-width below
        st.markdown("**Topic / Issue**")
        st.caption("Use the expander below to filter by topic.")

    # Topic filter — full 3-level hierarchy
    topic_active = topics_df[topics_df["value"] > 0].copy()
    selected_topic_categories: list = []
    selected_topic_issues: list = []
    selected_topic_subissues: list = []
    topic_strength = 1

    with st.expander("Topic / Issue filters", expanded=False):
        st.caption(
            "Select a category, then refine by issue and sub-issue. "
            "Lower branches inherit their parents."
        )
        selected_topic_categories = st.multiselect(
            "Category",
            options=sorted([v for v in topic_active["category"].dropna().unique() if v]),
        )

        issue_source = topic_active
        if selected_topic_categories:
            issue_source = issue_source[
                issue_source["category"].isin(selected_topic_categories)
            ]
        selected_topic_issues = st.multiselect(
            "Issue",
            options=sorted([v for v in issue_source["issue_label"].dropna().unique() if v]),
        )

        if selected_topic_issues:
            topic_strength = st.select_slider(
                "Issue-level minimum score",
                options=[1, 2, 3],
                value=1,
                format_func=format_topic_level_label,
                help="Applied only to issue-level topics. Sub-issues remain presence/absence.",
            )

        subissue_source = issue_source
        if selected_topic_issues:
            subissue_source = subissue_source[
                subissue_source["issue_label"].isin(selected_topic_issues)
            ]
        selected_topic_subissues = st.multiselect(
            "Sub-issue",
            options=sorted(
                [v for v in subissue_source["subissue_label"].dropna().unique() if v]
            ),
        )

    use_topic_scoped_text = st.checkbox(
        "Search only within topic-tagged text",
        value=False,
        disabled=not (
            selected_topic_categories or selected_topic_issues or selected_topic_subissues
        ),
        help=(
            "When topic filters are selected, search only the text segments that were tagged for those topics, "
            "instead of the full agreement text."
        ),
    )
    if use_topic_scoped_text:
        st.info(
            "Topic-scoped mode is active: the search will run only on text segments tagged to the selected topics."
        )

    # Apply metadata filters
    filtered_pax = pax_df.copy()
    if selected_regions:
        filtered_pax = filtered_pax[filtered_pax["Reg"].astype(str).isin(selected_regions)]
    if selected_conflicts:
        filtered_pax = filtered_pax[filtered_pax["Con"].astype(str).isin(selected_conflicts)]
    if selected_stages:
        filtered_pax = filtered_pax[
            filtered_pax["stage_label"].astype(str).isin(selected_stages)
        ]
    if selected_agt_types:
        filtered_pax = filtered_pax[
            filtered_pax["agt_type"].astype(str).isin(selected_agt_types)
        ]
    if year_range:
        filtered_pax = filtered_pax[
            filtered_pax["year"].between(year_range[0], year_range[1])
        ]

    # Apply topic filter
    if selected_topic_categories or selected_topic_issues or selected_topic_subissues:
        topic_mask = topic_active["value"] > 0
        if selected_topic_categories:
            topic_mask &= topic_active["category"].isin(selected_topic_categories)
        if selected_topic_issues:
            topic_mask &= topic_active["issue_label"].isin(selected_topic_issues)
            issue_level = topic_active["type"].astype(str).eq("issue")
            topic_mask &= (~issue_level) | (topic_active["value"] >= topic_strength)
        if selected_topic_subissues:
            topic_mask &= topic_active["subissue_label"].isin(selected_topic_subissues)
        matching_ids = topic_active.loc[topic_mask, "AgtId"].dropna().unique()
        filtered_pax = filtered_pax[filtered_pax["AgtId"].isin(matching_ids)]

    any_filter = any(
        [
            selected_regions,
            selected_conflicts,
            selected_stages,
            selected_agt_types,
            selected_topic_categories,
            selected_topic_issues,
            selected_topic_subissues,
            year_range and year_range != (year_values[0], year_values[-1])
            if year_values
            else False,
        ]
    )
    n_filtered = len(filtered_pax)

    if not any_filter:
        st.info(
            f"Searching the full corpus (~{n_filtered:,} agreements). "
            "Add a filter to speed things up and narrow results."
        )
    else:
        st.success(
            f"{n_filtered:,} agreement{'s' if n_filtered != 1 else ''} match your filters."
        )

    st.divider()

    # ── Search ────────────────────────────────────────────────────────────────
    st.markdown("### Search")
    st.info(
        "Enter one term or multiple comma-separated terms. Each term is searched independently, "
        "and results report both agreement-level matches and sentence-level mentions for review."
    )

    search_input = st.text_input(
        "Enter search terms (comma-separated):",
        placeholder="e.g. amnesty, pardon, immunity",
        help="Each term is searched independently. Separate multiple terms with commas.",
    )

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        use_boolean = st.checkbox(
            "Boolean search mode",
            help="AND: all terms must appear in the same sentence. "
            "NOT: first term present, remaining terms absent.",
        )
    boolean_mode = "OR"
    if use_boolean:
        with adv_col2:
            boolean_mode = st.selectbox("Boolean operator", ["OR", "AND", "NOT"])

    phrase_search = st.checkbox(
        "Exact phrase match",
        help="Treat each comma-separated entry as an exact phrase (case-insensitive).",
    )
    use_lemmas = st.checkbox(
        "Match word variants",
        help="Approximate lemma matching for single-word terms. For example, a search for internal can also match internally.",
    )

    with st.expander("Search tips", expanded=False):
        st.markdown(
            "- Separate terms with commas to compare multiple concepts in one run.\n"
            "- Use exact phrase match for multi-word expressions such as peace dividend or mineral wealth.\n"
            "- Use Boolean mode when you want terms to co-occur in the same sentence rather than anywhere in the same agreement.\n"
            "- Turn on Match word variants to catch basic single-word forms such as internal and internally.\n"
            "- Review sentence-level mentions below results to exclude false positives before export."
        )

    run_search = st.button("Execute Search", type="primary")

    # Session state initialisation
    for key, default in [
        ("search_has_run", False),
        ("search_results", []),
        ("search_snippets", {}),
        ("search_mentions", []),
        ("search_term_list", []),
        ("search_mentions_editor", pd.DataFrame()),
        ("search_signature", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if run_search:
        if not search_input.strip():
            st.warning("Please enter at least one search term.")
        else:
            term_list = [
                t.strip().strip('"') if phrase_search else t.strip()
                for t in search_input.split(",")
                if t.strip()
            ]
            filtered_corpus = corpus_df[corpus_df["AgtId"].isin(filtered_pax["AgtId"])]
            search_frame = filtered_corpus
            search_text_column = "Agreement text"

            if use_topic_scoped_text:
                topic_segments_filtered = topic_segments_df[
                    topic_segments_df["AgtId"].isin(filtered_pax["AgtId"])
                ].copy()

                topic_segments_filtered = topic_segments_filtered[
                    topic_segments_filtered["tagged_text"].astype(str).str.strip().ne("")
                ]

                if selected_topic_categories:
                    topic_segments_filtered = topic_segments_filtered[
                        topic_segments_filtered["category"].isin(selected_topic_categories)
                    ]
                if selected_topic_issues:
                    topic_segments_filtered = topic_segments_filtered[
                        topic_segments_filtered["issue_label"].isin(selected_topic_issues)
                    ]
                    issue_level = topic_segments_filtered["type"].astype(str).eq("issue")
                    topic_segments_filtered = topic_segments_filtered[
                        (~issue_level) | (topic_segments_filtered["value"] >= topic_strength)
                    ]
                if selected_topic_subissues:
                    topic_segments_filtered = topic_segments_filtered[
                        topic_segments_filtered["subissue_label"].isin(selected_topic_subissues)
                    ]

                search_frame = topic_segments_filtered
                search_text_column = "tagged_text"

            with st.spinner("Searching agreements..."):
                progress_bar = st.progress(0)
                start = time.time()
                results, snippets, mention_rows = search_corpus(
                    search_frame,
                    pax_df,
                    term_list,
                    use_boolean,
                    boolean_mode,
                    progress_bar,
                    text_column=search_text_column,
                    use_lemmas=use_lemmas,
                )
                elapsed = time.time() - start
                progress_bar.empty()

            st.session_state.update(
                {
                    "search_has_run": True,
                    "search_results": results,
                    "search_snippets": snippets,
                    "search_mentions": mention_rows,
                    "search_term_list": term_list,
                    "search_mentions_editor": pd.DataFrame(),
                    "search_signature": (
                        tuple(term_list),
                        tuple(result["AgtId"] for result in results),
                        len(mention_rows),
                    ),
                }
            )

            if results:
                st.success(
                    f"Found {count_label(len(results), 'agreement')} "
                    f"with matches across {count_label(n_filtered, 'agreement')} searched. "
                    f"({elapsed:.1f}s)"
                )
            else:
                st.warning(
                    "No matches found. Try broadening your terms or removing filters."
                )

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state["search_has_run"] and st.session_state["search_results"]:
        results = st.session_state["search_results"]
        snippets = st.session_state["search_snippets"]
        mention_rows = st.session_state.get("search_mentions", [])
        term_list = st.session_state["search_term_list"]

        if not mention_rows and results:
            mention_rows = build_mention_rows_from_results(results, snippets, term_list)
            st.session_state["search_mentions"] = mention_rows

        results_df = pd.DataFrame(results)
        mentions_df = pd.DataFrame(mention_rows)
        current_signature = (
            tuple(term_list),
            tuple(results_df["AgtId"].tolist()),
            len(mention_rows),
        )

        if st.session_state.get("search_signature") != current_signature:
            st.session_state["search_mentions_editor"] = pd.DataFrame()
            st.session_state["search_signature"] = current_signature

        if not mentions_df.empty and st.session_state["search_mentions_editor"].empty:
            st.session_state["search_mentions_editor"] = mentions_df.copy()

        editable_mentions_df = st.session_state["search_mentions_editor"].copy()
        if editable_mentions_df.empty:
            editable_mentions_df = pd.DataFrame(columns=mentions_df.columns.tolist())

        st.divider()
        st.markdown("### Results")

        overview_cols = st.columns(min(len(term_list) + 1, 4))
        with overview_cols[0]:
            st.metric("Agreements found", len(results_df))

        for index, term in enumerate(term_list, start=1):
            agreements_with_term = int((results_df[f'"{term}"'] > 0).sum()) if f'"{term}"' in results_df.columns else 0
            with overview_cols[index % len(overview_cols)]:
                st.metric(term, count_label(agreements_with_term, "agreement"))

        overview_text = ", ".join(
            f"{term} ({count_label(int((results_df[f'"{term}"'] > 0).sum()) if f'"{term}"' in results_df.columns else 0, 'agreement')})"
            for term in term_list
        )
        if overview_text:
            st.caption(f"Per-term breakdown: {overview_text}.")

        term_cols = [f'"{t}"' for t in term_list]
        agreement_display_cols = [
            "Agreement",
            "Year",
            "Stage",
            "Region",
            "Country / Conflict",
            "Total mentions",
        ] + [c for c in term_cols if c in results_df.columns] + [
            "PA-X Link",
            "PDF Link",
        ]
        agreement_display_cols = [c for c in agreement_display_cols if c in results_df.columns]

        st.dataframe(
            results_df[agreement_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "PA-X Link": st.column_config.LinkColumn("PA-X Link"),
                "PDF Link": st.column_config.LinkColumn("PDF Link"),
                "Year": st.column_config.NumberColumn("Year", format="%d"),
            },
        )

        st.markdown("### Mention Review")
        st.markdown("Review individual matched sentences below, then exclude mentions or leave notes before export.")
        st.caption(
            "Use Exclude to remove a sentence-level mention from the cleaned export. "
            "Notes are optional and useful for recording why a mention was removed or flagged."
        )

        mention_display_cols = [
            "Exclude",
            "Agreement",
            "Matched sentence",
            "Matched terms",
            "Year",
            "Stage",
            "Region",
            "Country / Conflict",
            "PA-X Link",
            "PDF Link",
            "Notes",
        ]
        mention_display_cols = [c for c in mention_display_cols if c in editable_mentions_df.columns]

        edited_mentions_df = st.data_editor(
            editable_mentions_df[mention_display_cols],
            use_container_width=True,
            hide_index=True,
            key="search_mentions_editor_widget",
            column_config={
                "Exclude": st.column_config.CheckboxColumn("Exclude"),
                "Matched sentence": st.column_config.TextColumn("Matched sentence", width="large"),
                "Matched terms": st.column_config.TextColumn("Matched terms"),
                "PA-X Link": st.column_config.LinkColumn("PA-X Link"),
                "PDF Link": st.column_config.LinkColumn("PDF Link"),
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "Notes": st.column_config.TextColumn("Notes"),
            },
        )
        editable_mentions_df.update(edited_mentions_df)
        st.session_state["search_mentions_editor"] = editable_mentions_df

        review_cols = st.columns(3)
        with review_cols[0]:
            st.metric("Mentions found", len(editable_mentions_df))
        with review_cols[1]:
            excluded_mentions = int(editable_mentions_df["Exclude"].sum()) if "Exclude" in editable_mentions_df.columns else 0
            st.metric("Excluded mentions", excluded_mentions)
        with review_cols[2]:
            if "Exclude" in editable_mentions_df.columns:
                export_ready = editable_mentions_df[
                    editable_mentions_df["Exclude"] == False
                ]
            else:
                export_ready = editable_mentions_df.copy()
            st.metric("Mentions kept for export", len(export_ready))

        # Export
        export_df = editable_mentions_df.copy()
        st.download_button(
            "Download reviewed results as CSV",
            export_df.to_csv(index=False).encode("utf-8"),
            "pax_corpus_search_results.csv",
            "text/csv",
        )
        if "Exclude" in export_df.columns:
            corrected_export_df = export_df[
                export_df["Exclude"] == False
            ].copy()
        else:
            corrected_export_df = export_df.copy()
        st.download_button(
            "Download kept mentions CSV",
            corrected_export_df.to_csv(index=False).encode("utf-8"),
            "pax_corpus_search_results_corrected.csv",
            "text/csv",
        )

        # Per-agreement expandable detail
        st.markdown("### Agreement Detail")
        for row in results:
            agt_id = row["AgtId"]
            title = row["Agreement"]
            total = row["Total mentions"]
            year = row.get("Year", "")
            year_str = str(int(year)) if pd.notna(year) and year != "" else "N/A"
            label = (
                f"{title} ({year_str}) — "
                f"{total} mention{'s' if total != 1 else ''}"
            )
            with st.expander(label):
                st.markdown(
                    f"<div class='agreement-title'>"
                    f"{html_module.escape(str(title))}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                toolbar = (
                    "<div class='corpus-toolbar'>"
                    + render_link_button("Open on PA-X", row.get("PA-X Link", ""))
                    + render_link_button(
                        "Open PDF", row.get("PDF Link", ""), secondary=True
                    )
                    + "</div>"
                )
                st.markdown(toolbar, unsafe_allow_html=True)
                meta_parts = [
                    f"Stage: {row.get('Stage', '')}",
                    f"Region: {row.get('Region', '')}",
                    f"Country/Conflict: {row.get('Country / Conflict', '')}",
                ]
                st.caption(
                    " | ".join(p for p in meta_parts if p.split(": ", 1)[1])
                )
                for s in snippets.get(agt_id, []):
                    st.markdown(
                        f"<div class='snippet-block'>{s}</div>",
                        unsafe_allow_html=True,
                    )

        # ── Visualisations ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### Visualisations")

        visual_term_options = ["All terms"] + term_list
        selected_visual_term = st.selectbox(
            "Chart results for",
            visual_term_options,
            help="View charts for all matched agreements, or only agreements matching one selected search term.",
        )

        chart_df = results_df.copy()
        chart_title_suffix = "All terms"
        if selected_visual_term != "All terms":
            term_column = f'"{selected_visual_term}"'
            if term_column in chart_df.columns:
                chart_df = chart_df[chart_df[term_column] > 0].copy()
            chart_title_suffix = selected_visual_term

        if chart_df.empty:
            st.info("No chart data is available for the selected term filter.")
        else:
            vc1, vc2, vc3 = st.columns(3)

            with vc1:
                year_data = (
                    chart_df.dropna(subset=["Year"])
                    .assign(Year=lambda d: d["Year"].astype(int))
                    .groupby("Year")
                    .size()
                    .reset_index(name="Agreements")
                )
                fig_year = px.bar(
                    year_data,
                    x="Year",
                    y="Agreements",
                    title=f"Agreements by Year: {chart_title_suffix}",
                    color_discrete_sequence=["#091f40"],
                )
                fig_year.update_layout(
                    font_family="Montserrat",
                    xaxis_title="Year",
                    yaxis_title="Agreements",
                )
                st.plotly_chart(fig_year, use_container_width=True)
                st.download_button(
                    "Download (CSV)",
                    year_data.to_csv(index=False).encode("utf-8"),
                    "by_year.csv",
                    "text/csv",
                    key="dl_year",
                )

            with vc2:
                stage_data = (
                    chart_df.groupby("Stage").size().reset_index(name="Agreements")
                )
                fig_stage = px.bar(
                    stage_data,
                    x="Stage",
                    y="Agreements",
                    title=f"Agreements by Stage: {chart_title_suffix}",
                    color_discrete_sequence=["#3a7fc1"],
                )
                fig_stage.update_layout(
                    font_family="Montserrat",
                    xaxis_tickangle=-30,
                    xaxis_title="Stage",
                    yaxis_title="Agreements",
                )
                st.plotly_chart(fig_stage, use_container_width=True)
                st.download_button(
                    "Download (CSV)",
                    stage_data.to_csv(index=False).encode("utf-8"),
                    "by_stage.csv",
                    "text/csv",
                    key="dl_stage",
                )

            with vc3:
                region_data = (
                    chart_df.groupby("Region").size().reset_index(name="Agreements")
                )
                fig_region = px.bar(
                    region_data,
                    x="Region",
                    y="Agreements",
                    title=f"Agreements by Region: {chart_title_suffix}",
                    color_discrete_sequence=["#1a6b4a"],
                )
                fig_region.update_layout(
                    font_family="Montserrat",
                    xaxis_tickangle=-30,
                    xaxis_title="Region",
                    yaxis_title="Agreements",
                )
                st.plotly_chart(fig_region, use_container_width=True)
                st.download_button(
                    "Download (CSV)",
                    region_data.to_csv(index=False).encode("utf-8"),
                    "by_region.csv",
                    "text/csv",
                    key="dl_region",
                )

        # N-gram analysis (collapsed)
        with st.expander("N-gram analysis (experimental)", expanded=False):
            from collections import Counter

            st.markdown(
                "An n-gram is a short sequence of adjacent words. A 2-gram is a bigram, "
                "such as mineral resources, and a 3-gram is a trigram, such as conflict mineral trade."
            )
            st.markdown(
                "This section counts the most common word sequences inside the matched sentences only. "
                "It helps show recurring language patterns around your search terms, which can be useful "
                "for exploratory analysis, query refinement, or spotting common frames and formulations in the corpus."
            )
            st.caption(
                "Interpret with care: n-grams show frequent phrasing, not meaning or causal importance. "
                "They are best used as a quick descriptive summary of the matched text."
            )

            ngram_n = st.number_input(
                "N-gram size", min_value=2, max_value=5, value=2, key="ngram_n",
                help="Choose how many adjacent words to count together. 2 = bigrams, 3 = trigrams.",
            )
            if st.button("Generate N-grams", key="btn_ngram"):
                ngrams: Counter = Counter()
                for aid, slist in snippets.items():
                    for s in slist:
                        plain = re.sub(r"<[^>]+>", "", s)
                        tokens = plain.lower().split()
                        for j in range(len(tokens) - ngram_n + 1):
                            ngrams[tuple(tokens[j : j + ngram_n])] += 1
                top = ngrams.most_common(20)
                ng_df = pd.DataFrame(
                    [(" ".join(k), v) for k, v in top], columns=["N-gram", "Count"]
                )
                st.dataframe(ng_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
