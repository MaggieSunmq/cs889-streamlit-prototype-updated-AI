import json
import os
import streamlit as st
from google import genai

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Literature Search — AI", layout="wide")

# -----------------------------
# Config
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "example-bib.json")
MODEL_NAME = "gemini-2.5-flash"

HAS_GEMINI = bool(os.getenv("GEMINI_API_KEY"))
client = genai.Client() if HAS_GEMINI else None


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_papers(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["references"]


def norm_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [str(v)]


def norm_authors(p):
    return [str(x) for x in norm_list(p.get("authors")) if str(x).strip()]


def norm_keywords(p):
    return [str(x) for x in norm_list(p.get("keywords")) if str(x).strip()]


def paper_url(p):
    url = p.get("url") or p.get("link") or p.get("pdf")
    if url:
        return str(url).strip()
    doi = (p.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return ""


def brief_for_ai(p, abstract_chars=420):
    abstract = (p.get("abstract") or "").replace("\n", " ").strip()
    if len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars] + "…"
    return {
        "id": p.get("id"),
        "title": p.get("title", ""),
        "year": p.get("year"),
        "authors": norm_authors(p)[:10],
        "journal": p.get("journal", "") or p.get("venue", ""),
        "keywords": norm_keywords(p)[:15],
        "doi": p.get("doi", ""),
        "url": paper_url(p),
        "abstract": abstract,
    }


def parse_json_lenient(text: str):
    if not text:
        return None
    t = text.strip()

    if t.startswith("```"):
        t = t.strip("`").strip()
        lines = t.splitlines()
        if lines and lines[0].strip().lower() in ("json", "javascript"):
            t = "\n".join(lines[1:]).strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return None
    return None


def toggle_save(pid, saved_ids):
    if pid is None:
        return
    if pid in saved_ids:
        saved_ids.discard(pid)
    else:
        saved_ids.add(pid)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
div.stButton > button{
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
}
.meta-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin: 0.1rem 0.25rem 0.1rem 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 0.85rem;
    opacity: 0.95;
}
.meta-kv {
    margin: 0.15rem 0 0.35rem 0;
    font-size: 0.95rem;
}
.meta-k {
    opacity: 0.7;
    display: inline-block;
    min-width: 90px;
}
.meta-v {
    opacity: 0.98;
}
.small-muted {
    opacity: 0.7;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_metadata_pretty_no_columns(p, key_prefix=""):
    pid = p.get("id")
    year = p.get("year")
    venue = p.get("journal", "") or p.get("venue", "")
    authors = norm_authors(p)
    doi = (p.get("doi") or "").strip()
    url = paper_url(p)
    keywords = norm_keywords(p)

    def kv(k, v):
        vv = v if (v is not None and str(v).strip()) else "—"
        st.markdown(
            f'<div class="meta-kv"><span class="meta-k">{k}</span>'
            f'<span class="meta-v">{vv}</span></div>',
            unsafe_allow_html=True,
        )

    kv("Year", year)
    kv("Venue", venue)
    kv("DOI", doi)
    kv("URL", url)
    kv("ID", pid if pid is not None else "")

    st.markdown('<div class="meta-kv"><span class="meta-k">Authors</span></div>', unsafe_allow_html=True)
    if authors:
        if len(authors) <= 12:
            st.write(", ".join(authors))
        else:
            st.write(", ".join(authors[:12]) + f" … (+{len(authors)-12} more)")
    else:
        st.write("—")

    st.markdown('<div class="meta-kv"><span class="meta-k">Keywords</span></div>', unsafe_allow_html=True)
    if keywords:
        chips = " ".join([f'<span class="meta-chip">{k}</span>' for k in keywords[:30]])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("—")

    show_raw = st.toggle("Show raw JSON", value=False, key=f"{key_prefix}raw_{pid}")
    if show_raw:
        st.json(p)


def summarize_abstract_with_gemini(title: str, abstract: str) -> str:
    prompt = f"""
You summarize academic papers using ONLY the abstract.

Title: {title}

Abstract:
{abstract}

Return:
- 1 sentence plain-English takeaway
- 3 bullet points: method / data / results (if available)
- 5 keywords

Be faithful to the abstract; do not add facts not stated.
"""
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return (resp.text or "").strip()


def render_paper_card_ai(p, saved_ids, key_prefix=""):
    pid = p.get("id")
    title = p.get("title", "(no title)")
    year = p.get("year", "")
    venue = p.get("journal", "") or p.get("venue", "")
    authors = ", ".join(norm_authors(p))
    url = paper_url(p)
    doi = (p.get("doi") or "").strip()
    kw = norm_keywords(p)
    abstract = (p.get("abstract") or "").strip()

    saved = pid in saved_ids
    star = "★" if saved else "☆"
    save_text = "Saved" if saved else "Save"

    cols = st.columns([0.12, 0.88], vertical_alignment="top")
    with cols[0]:
        btn_key = f"{key_prefix}save_{pid}_{hash(title)}"
        if st.button(f"{star} {save_text}", key=btn_key, use_container_width=True):
            toggle_save(pid, saved_ids)
            st.rerun()

    with cols[1]:
        st.markdown(f"**{title}**")
        if year or venue:
            st.caption(f"{year} • {venue}")
        if authors:
            st.write(authors)

        if url:
            st.write(url)
        if doi:
            st.write(f"**DOI:** {doi}")
        if kw:
            preview = ", ".join(kw[:12])
            more = f" … (+{len(kw)-12})" if len(kw) > 12 else ""
            st.caption("Tags: " + preview + more)

        tab_meta, tab_abs, tab_sum = st.tabs(["Metadata", "Abstract", "Summary"])
        with tab_meta:
            render_metadata_pretty_no_columns(p, key_prefix=key_prefix)

        with tab_abs:
            if abstract:
                st.write(abstract)
            else:
                st.caption("No abstract.")

        with tab_sum:
            if not abstract:
                st.caption("No abstract available, so there is nothing to summarize.")
            elif not HAS_GEMINI:
                st.warning("Gemini is not configured (missing GEMINI_API_KEY). Add it in Streamlit Cloud secrets.")
            else:
                if "summaries" not in st.session_state:
                    st.session_state.summaries = {}
                cache_key = str(pid) if pid is not None else f"{hash(title)}"

                if cache_key in st.session_state.summaries:
                    st.markdown(st.session_state.summaries[cache_key])
                    if st.button("Regenerate summary", key=f"{key_prefix}regen_{cache_key}"):
                        with st.spinner("Summarizing…"):
                            s = summarize_abstract_with_gemini(title, abstract)
                        st.session_state.summaries[cache_key] = s
                        st.rerun()
                else:
                    st.markdown(
                        '<div class="small-muted">Generate a summary from the abstract.</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Summarize with Gemini", type="primary", key=f"{key_prefix}sum_{cache_key}"):
                        with st.spinner("Summarizing…"):
                            s = summarize_abstract_with_gemini(title, abstract)
                        st.session_state.summaries[cache_key] = s
                        st.rerun()

    st.divider()


def run_ai_retrieval(intent: str, top_k_ai: int, papers, papers_by_id):
    library = [brief_for_ai(p) for p in papers]
    prompt = f"""
You are an AI retrieval tool over a local paper library.

Return ONLY valid JSON with exactly this schema:
{{
  "paper_ids": ["... up to {top_k_ai} ids from the library ..."],
  "note": "1-2 sentence rationale"
}}

User intent: {intent}

Library (JSON list):
{json.dumps(library, ensure_ascii=False)}
"""
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    parsed = parse_json_lenient(resp.text or "") or {}
    ids = parsed.get("paper_ids", [])
    if not isinstance(ids, list):
        ids = []
    ids = [pid for pid in ids if pid in papers_by_id]
    note = str(parsed.get("note", "")).strip()
    return ids, note


def chat_planner(user_msg: str, total_papers: int):
    prompt = f"""
You are a research assistant helping a user search papers in a LOCAL library (size: {total_papers}).

Return ONLY valid JSON:
{{
  "assistant_message": "short helpful message (2-5 sentences)",
  "keyword_query": "a compact keyword-style query for lexical matching",
  "ai_intent": "a richer natural-language intent for AI retrieval",
  "filters": {{
    "only_with_doi": true/false,
    "year_min": number or null,
    "year_max": number or null
  }}
}}

Rules:
- Keep keyword_query short and specific (no long sentences).
- ai_intent can be more descriptive.
- If the user didn’t mention DOI/year constraints, set filters to null/false.
- Do not mention web search. This is local-only.
- Output JSON only.

User message:
{user_msg}
"""
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return parse_json_lenient(resp.text or "")


def render_saved_panel(papers_by_id):
    st.markdown("---")
    st.markdown(f"### Saved papers ({len(st.session_state.saved_ids)})")

    saved = [papers_by_id[pid] for pid in st.session_state.saved_ids if pid in papers_by_id]
    saved.sort(key=lambda p: (p.get("year") is None, p.get("year", 0)), reverse=True)

    if not saved:
        st.caption("No saved papers yet. Click ☆ Save on any paper card to save it.")
        return

    a1, a2, a3 = st.columns([0.34, 0.33, 0.33], vertical_alignment="center")
    with a1:
        export_obj = {"references": saved}
        st.download_button(
            "Download saved papers (JSON)",
            data=json.dumps(export_obj, ensure_ascii=False, indent=2),
            file_name="saved-papers.json",
            mime="application/json",
            key="download_saved",
            use_container_width=True,
        )
    with a2:
        if st.button("Clear all saved", use_container_width=True):
            st.session_state.saved_ids = set()
            st.rerun()
    with a3:
        st.caption("Saved items persist during this session.")

    for i, p in enumerate(saved):
        render_paper_card_ai(p, st.session_state.saved_ids, key_prefix=f"saved_{i}_")


# -----------------------------
# Load data & session state
# -----------------------------
papers = load_papers()
papers_by_id = {p.get("id"): p for p in papers if p.get("id") is not None}

if "saved_ids" not in st.session_state:
    st.session_state.saved_ids = set()
if "query" not in st.session_state:
    st.session_state.query = ""
if "ai_selected_ids" not in st.session_state:
    st.session_state.ai_selected_ids = []
if "ai_note" not in st.session_state:
    st.session_state.ai_note = ""
if "ai_intent_override" not in st.session_state:
    st.session_state.ai_intent_override = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_suggestions" not in st.session_state:
    st.session_state.chat_suggestions = {
        "assistant_message": "",
        "keyword_query": "",
        "ai_intent": "",
        "filters": {"only_with_doi": False, "year_min": None, "year_max": None},
    }

# -----------------------------
# UI
# -----------------------------
st.markdown("## Literature Search — AI")

with st.expander("Chat assistant (optional) — refine your search goal", expanded=False):
    if not HAS_GEMINI:
        st.warning("Gemini is not configured (missing GEMINI_API_KEY). Add it in Streamlit Cloud secrets to use chat.")
    else:
        bar_left, bar_right = st.columns([0.7, 0.3], vertical_alignment="center")
        with bar_left:
            st.caption("Chat history is stored only for this session.")
        with bar_right:
            if st.button("Clear chat history", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.chat_suggestions = {
                    "assistant_message": "",
                    "keyword_query": "",
                    "ai_intent": "",
                    "filters": {"only_with_doi": False, "year_min": None, "year_max": None},
                }
                st.rerun()

        st.divider()

        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        auto_run_ai_after_chat = st.checkbox("Auto-run AI retrieval after chat", value=False)

        user_msg = st.chat_input("Describe what you want (topic, methods, constraints, etc.)")
        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.spinner("Thinking…"):
                parsed = chat_planner(user_msg, total_papers=len(papers)) or {}

            assistant_message = str(parsed.get("assistant_message", "")).strip()
            ai_intent_from_chat = str(parsed.get("ai_intent", "")).strip()

            st.session_state.chat_suggestions = {
                "assistant_message": assistant_message,
                "keyword_query": str(parsed.get("keyword_query", "")).strip(),
                "ai_intent": ai_intent_from_chat,
                "filters": parsed.get("filters", {"only_with_doi": False, "year_min": None, "year_max": None}),
            }

            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message or "(no message)"})
            with st.chat_message("assistant"):
                st.markdown(assistant_message or "(no message)")

            if auto_run_ai_after_chat and ai_intent_from_chat:
                with st.spinner("Auto-running AI retrieval…"):
                    ids, note = run_ai_retrieval(ai_intent_from_chat, top_k_ai=20, papers=papers, papers_by_id=papers_by_id)
                st.session_state.ai_selected_ids = ids
                st.session_state.ai_note = note
                st.session_state.ai_intent_override = ai_intent_from_chat
                st.rerun()

        sugg = st.session_state.chat_suggestions
        st.markdown("---")
        st.markdown("**Suggested queries from chat**")
        st.write(f"**Keyword query:** {sugg.get('keyword_query') or '—'}")
        st.write(f"**AI intent:** {sugg.get('ai_intent') or '—'}")

# AI controls
c1, c2, c3, c4 = st.columns([0.55, 0.15, 0.15, 0.15], vertical_alignment="bottom")
with c1:
    st.text_input(
        "Main query (optional)",
        key="query",
        placeholder="Type a short phrase",
    )
with c2:
    top_k_ai = st.slider("AI results", 5, 50, 20)
with c3:
    use_same_query = st.checkbox("AI uses same query", value=True)
with c4:
    pass  # spacer

q = (st.session_state.query or "").strip()
if use_same_query:
    ai_intent = q
    st.caption("AI intent is the same as the main query.")
else:
    ai_intent = st.text_input(
        "AI intent (optional override)",
        value=st.session_state.ai_intent_override or q,
        placeholder="Describe what you want (e.g., causal probing for interpretability)",
        key="ai_intent_override",
    )

b1, b2 = st.columns([0.5, 0.5])
with b1:
    run_ai = st.button("Run AI search", type="primary", use_container_width=True)
with b2:
    clear_ai = st.button("Clear AI results", use_container_width=True)

if clear_ai:
    st.session_state.ai_selected_ids = []
    st.session_state.ai_note = ""
    st.rerun()

if run_ai:
    if not ai_intent.strip():
        st.warning("Please enter a query first.")
    elif not HAS_GEMINI:
        st.warning("Gemini is not configured (missing GEMINI_API_KEY). Add it in Streamlit Cloud secrets.")
    else:
        with st.spinner("Selecting papers…"):
            ids, note = run_ai_retrieval(ai_intent, top_k_ai=top_k_ai, papers=papers, papers_by_id=papers_by_id)
        st.session_state.ai_selected_ids = ids
        st.session_state.ai_note = note
        st.rerun()

if st.session_state.ai_note:
    st.caption(st.session_state.ai_note)

if not st.session_state.ai_selected_ids:
    st.caption(f"Found 0 of {len(papers)} papers. Click “Run AI search” to select papers.")
else:
    st.caption(f"Found {len(st.session_state.ai_selected_ids)} of {len(papers)} papers.")
    with st.container(height=650):
        for i, pid in enumerate(st.session_state.ai_selected_ids):
            render_paper_card_ai(papers_by_id[pid], st.session_state.saved_ids, key_prefix=f"ai_{i}_")

render_saved_panel(papers_by_id)

