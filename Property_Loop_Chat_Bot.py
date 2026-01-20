
import os
import re
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# Embedding model (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Missing dependency: sentence-transformers. Install with `pip install sentence-transformers`.")
    raise

# FAISS
try:
    import faiss
except Exception as e:
    st.error("Missing dependency: faiss-cpu. Install with `pip install faiss-cpu`.")
    raise

# We will import transformers and torch only when the user enables LLM to avoid heavy imports up front.

# -------------------------
# Config
# -------------------------
NOT_FOUND = "Sorry can not find the answer"
DEFAULT_HOLDINGS_PATH = r"holdings.csv"
DEFAULT_TRADES_PATH = r"trades.csv"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

EMBEDDINGS_CACHE = "embeddings.npy"
DOCUMENTS_CACHE = "documents.npy"
FAISS_INDEX_PATH = "index.faiss"
METADATA_CACHE = "metadatas.npy"

DEFAULT_TOP_K = 5
DEFAULT_SIM_THRESHOLD = 0.35
BATCH = 256

# -------------------------
# Utility helpers
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def normalize_series_names(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    if df is None:
        return None
    cols = df.columns.tolist()
    lower = {c.lower(): c for c in cols}
    for cand in filter(None, candidates):
        if cand.lower() in lower:
            return lower[cand.lower()]
    for key, original in lower.items():
        for cand in filter(None, candidates):
            if cand.lower() in key:
                return original
    return None


def find_best_matching_col_value(df: pd.DataFrame, col: str, target: str) -> Tuple[pd.Series, bool]:
    if col is None or col not in df.columns:
        return pd.Series([False] * len(df), index=df.index), False
    series_norm = normalize_series_names(df[col])
    t = str(target).strip().lower()
    exact_mask = series_norm == t
    if exact_mask.sum() > 0:
        return exact_mask, True
    contains_mask = series_norm.str.contains(re.escape(t), na=False)
    if contains_mask.sum() > 0:
        return contains_mask, False
    tokens = [tok for tok in re.split(r"\W+", t) if len(tok) > 2]
    if tokens:
        def token_check(x: str) -> bool:
            if not x:
                return False
            parts = x.split()
            return all(any(tok in part for part in parts) for tok in tokens)
        token_mask = series_norm.apply(token_check)
        if token_mask.sum() > 0:
            return token_mask, False
    return pd.Series([False] * len(df), index=df.index), False


def row_to_text(row: pd.Series, prefix: str) -> str:
    parts = []
    for k, v in row.items():
        val = "" if pd.isna(v) else str(v)
        parts.append(f"{k}: {val}")
    return prefix + " | " + " ; ".join(parts)


# -------------------------
# Deterministic analytics (fast)
# -------------------------
def detect_fund_column(holdings_df: pd.DataFrame):
    candidates = ["PortfolioName", "portfolioName", "Portfolio", "Fund", "fund", "ShortName"]
    for name in candidates:
        if name in holdings_df.columns:
            return name
    return find_column(holdings_df, candidates)


def detect_pl_column(holdings_df: pd.DataFrame, trades_df: pd.DataFrame):
    pl_candidates = ["PL_YTD", "PL_MTD", "PL_QTD", "PL_DTD", "PL", "P&L", "Pnl", "PNL", "Profit", "profit"]
    for c in pl_candidates:
        if c in holdings_df.columns:
            return c
    combined = pd.concat([holdings_df, trades_df], ignore_index=True)
    return find_column(combined, pl_candidates)


def total_trades_for_fund(trades_df: pd.DataFrame, fund_name: str, fund_col_name) -> str:
    if fund_col_name is None:
        return NOT_FOUND
    mask, exact = find_best_matching_col_value(trades_df, fund_col_name, fund_name)
    count = int(mask.sum())
    if count == 0:
        return NOT_FOUND
    if exact:
        return f"Total trades for fund '{fund_name}': {count}"
    return f"Total trades for fund '{fund_name}' (approx. match): {count}"


def total_holdings_for_fund(holdings_df: pd.DataFrame, fund_name: str, fund_col_name) -> str:
    if fund_col_name is None:
        return NOT_FOUND
    mask, exact = find_best_matching_col_value(holdings_df, fund_col_name, fund_name)
    count = int(mask.sum())
    if count == 0:
        return NOT_FOUND
    if exact:
        return f"Total holdings for fund '{fund_name}': {count}"
    return f"Total holdings for fund '{fund_name}' (approx. match): {count}"


def yearly_pnl_by_fund_df(holdings_df: pd.DataFrame, trades_df: pd.DataFrame, pl_col: str, fund_col_name: str):
    combined = pd.concat([holdings_df, trades_df], ignore_index=True)
    date_col = find_column(combined, ["AsOfDate", "TradeDate", "date", "Trade_Date", "SettleDate"])
    if pl_col is None or date_col is None:
        raise ValueError("P&L or date column not found; cannot compute yearly P&L")
    df = combined.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    fund_col = find_column(df, [fund_col_name, "PortfolioName", "portfolioName"])
    if fund_col is None:
        raise ValueError("Fund column not found in combined data")
    df[pl_col] = pd.to_numeric(df.get(pl_col, 0), errors="coerce").fillna(0.0)
    agg = df.groupby(["year", fund_col], as_index=False)[pl_col].sum()
    return agg


def best_worst_fund_by_year(holdings_df: pd.DataFrame, trades_df: pd.DataFrame, pl_col: str, year: int, fund_col_name: str):
    try:
        agg = yearly_pnl_by_fund_df(holdings_df, trades_df, pl_col, fund_col_name)
    except Exception:
        return NOT_FOUND
    df_year = agg[agg["year"] == int(year)]
    if df_year.empty:
        return NOT_FOUND
    fund_col = find_column(df_year, [fund_col_name, "PortfolioName", "portfolioName"])
    best_row = df_year.loc[df_year[pl_col].idxmax()]
    worst_row = df_year.loc[df_year[pl_col].idxmin()]
    return (
        f"Best performing fund in {year}: {best_row[fund_col]} with P&L {best_row[pl_col]}\n"
        f"Worst performing fund in {year}: {worst_row[fund_col]} with P&L {worst_row[pl_col]}"
    )


def top_holdings_for_fund(holdings_df: pd.DataFrame, fund_name: str, fund_col_name, by: str = "MV_Local", n: int = 10) -> str:
    if fund_col_name is None:
        return NOT_FOUND
    mv_col = find_column(holdings_df, [by, "MV_Local", "MV_Base", "Qty", "StartQty"])
    if mv_col is None:
        return NOT_FOUND
    mask, exact = find_best_matching_col_value(holdings_df, fund_col_name, fund_name)
    df_sel = holdings_df[mask].copy()
    if df_sel.empty:
        return NOT_FOUND
    df_sel[mv_col] = pd.to_numeric(df_sel.get(mv_col, 0), errors="coerce").fillna(0.0)
    df_sorted = df_sel.sort_values(by=mv_col, ascending=False).head(n)
    sec_col = find_column(df_sorted, ["SecName", "ShortName", "Name"])
    rows = []
    for _, r in df_sorted.iterrows():
        sec = r.get(sec_col, "") if sec_col else ""
        val = r.get(mv_col, 0)
        rows.append(f"{sec} — {mv_col}: {val}")
    header = f"Top {len(rows)} holdings for '{fund_name}':"
    return header + "\n" + "\n".join(rows)


# -------------------------
# Embedding & FAISS utilities
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(model_name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)


def _compute_embeddings_in_batches(embed_model: SentenceTransformer, documents: List[str]):
    emb_batches = []
    for i in range(0, len(documents), BATCH):
        batch = documents[i:i + BATCH]
        emb = embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        emb_batches.append(emb)
    embeddings = np.vstack(emb_batches).astype("float32")
    return embeddings


# Note: use leading underscore for embed model param to prevent Streamlit hashing errors
@st.cache_resource(show_spinner=False)
def build_or_load_faiss_index(documents: List[str], _embed_model: SentenceTransformer, rebuild: bool = False):
    # Try load caches if not rebuilding
    if not rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(DOCUMENTS_CACHE):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            embeddings = np.load(EMBEDDINGS_CACHE)
            documents_cached = list(np.load(DOCUMENTS_CACHE, allow_pickle=True))
            if len(documents_cached) != len(documents):
                st.info("Document count differs from cache; rebuilding index.")
                raise RuntimeError("doc count mismatch")
            return index, embeddings
        except Exception:
            st.info("Cache invalid or incompatible; rebuilding index.")

    embeddings = _compute_embeddings_in_batches(_embed_model, documents)

    dim = embeddings.shape[1]
    n_docs = embeddings.shape[0]
    if n_docs < 512:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        nlist = min(256, max(64, n_docs // 100))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(16, max(1, nlist // 8))

    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(EMBEDDINGS_CACHE, embeddings)
        np.save(DOCUMENTS_CACHE, np.array(documents, dtype=object), allow_pickle=True)
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

    return index, embeddings


# -------------------------
# Optional LLM loader (Qwen 0.5B) - loads only when requested
# -------------------------
@st.cache_resource(show_spinner=False)
def load_llm_pipeline(model_name: str):
    """
    Returns a text-generation pipeline or None on failure.
    This imports transformers and torch lazily.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except Exception as e:
        st.error(f"Required packages for LLM are missing: {e}")
        return None

    try:
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                       max_new_tokens=256, do_sample=False, device=device)
        return gen
    except Exception as e:
        st.error(f"Failed to load LLM '{model_name}': {e}")
        return None


# -------------------------
# RAG ask_bot
# -------------------------
ANALYTIC_KEYWORDS = [
    "total", "count", "how many", "best performing", "worst performing",
    "yearly", "profit", "p&l", "pnl", "holdings", "trades", "top holdings", "top"
]


def looks_analytic(q: str) -> bool:
    return any(kw in q.lower() for kw in ANALYTIC_KEYWORDS)


def build_prompt(context_docs: List[str], query: str) -> str:
    context_text = "\n\n".join(context_docs)
    prompt = (
        "You are an assistant that MUST answer using ONLY the provided CONTEXT and MUST NOT repeat the CONTEXT verbatim.\n"
        "If the answer cannot be found verbatim or by direct inference from the context, reply exactly: Sorry can not find the answer\n\n"
        "CONTEXT (DO NOT REPEAT):\n"
        + context_text
        + "\n\nQUESTION: "
        + query
        + "\n\nAnswer concisely using only the context."
    )
    return prompt


def ask_bot(query: str,
            index,
            documents: List[str],
            metadatas: List[Dict[str, Any]],
            embed_model: SentenceTransformer,
            holdings_df: pd.DataFrame,
            trades_df: pd.DataFrame,
            pl_col: str,
            fund_col_name: str,
            llm_pipeline=None,
            use_llm: bool = False,
            top_k: int = DEFAULT_TOP_K,
            sim_threshold: float = DEFAULT_SIM_THRESHOLD) -> Tuple[str, List[Tuple[str, float, str]]]:
    # Deterministic shortcuts
    try:
        if looks_analytic(query):
            ql = query.lower()
            m_top = re.search(r"(?:top|largest).*in\s+(?:fund\s+)?'?(?P<name>[^']+)'?", ql)
            if m_top:
                name = m_top.group("name").strip()
                return top_holdings_for_fund(holdings_df, name, fund_col_name, by="MV_Local", n=10), []
            m = re.search(r"total\s+trades\s+for\s+(?:fund\s+)?'?(?P<name>[^']+)'?", ql)
            if m:
                name = m.group("name").strip()
                return total_trades_for_fund(trades_df, name, fund_col_name), []
            m2 = re.search(r"total\s+holdings\s+for\s+(?:fund\s+)?'?(?P<name>[^']+)'?", ql)
            if m2:
                name = m2.group("name").strip()
                return total_holdings_for_fund(holdings_df, name, fund_col_name), []
            m3 = re.search(r"best\s+performing\s+fund\s+in\s+(?P<year>\d{4})", ql)
            if m3:
                return best_worst_fund_by_year(holdings_df, trades_df, pl_col, int(m3.group("year")), fund_col_name), []
            if "year" in ql or "yearly" in ql or "p&l" in ql or "pnl" in ql or "profit" in ql:
                try:
                    df = yearly_pnl_by_fund_df(holdings_df, trades_df, pl_col, fund_col_name)
                    s = df.to_string(index=False)
                    return "Yearly P&L by fund:\n" + s, []
                except Exception:
                    return NOT_FOUND, []
    except Exception:
        pass

    # Retrieval
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, ids = index.search(q_emb, top_k)
    sims = sims[0]
    ids = ids[0]
    results = []
    contexts = []
    for idx, sim in zip(ids, sims):
        if idx < 0:
            continue
        doc = documents[int(idx)]
        md = metadatas[int(idx)]
        results.append((doc, float(sim), md.get("source", "")))
        contexts.append((doc, float(sim), md.get("source", "")))

    if len(results) == 0:
        return NOT_FOUND, contexts
    max_sim = max([r[1] for r in results])
    if max_sim < sim_threshold:
        return NOT_FOUND, contexts

    # If LLM disabled, return short snippet from top context (fast)
    if not use_llm or llm_pipeline is None:
        top_doc, top_sim, top_src = results[0]
        snippet = top_doc[:400]
        return snippet, contexts

    # Build prompt and query LLM
    context_texts = [r[0] for r in results[:top_k]]
    prompt = build_prompt(context_texts, query)
    out = llm_pipeline(prompt)
    if isinstance(out, list) and len(out) > 0:
        gen = out[0].get("generated_text", "")
    else:
        gen = str(out)

    text = gen
    if isinstance(text, str) and text.startswith(prompt):
        text = text[len(prompt):].strip()
    else:
        for ct in context_texts:
            if ct and ct in text:
                text = text.replace(ct, "")
        text = text.strip()

    if not text:
        return NOT_FOUND, contexts

    ctx_concat = " ".join(context_texts).lower()
    tokens_ctx = re.findall(r"\w{4,}", ctx_concat)[:40]
    match_count = sum(1 for t in tokens_ctx if t in text.lower())
    if match_count == 0 and max_sim < 0.65:
        return NOT_FOUND, contexts

    text_lower = text.lower()
    if text_lower.startswith("you are an assistant") or ("context" in text_lower and len(text) > 400):
        return NOT_FOUND, contexts

    return text.strip(), contexts


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PropertyLoop RAG Chatbot (Qwen)", layout="wide")
st.title("PropertyLoop RAG Chatbot — Qwen ")

with st.sidebar:
    st.markdown("### Data input")
    use_upload = st.checkbox("Upload CSVs (otherwise use local defaults)", value=False)
    if use_upload:
        holdings_file = st.file_uploader("Upload holdings CSV", type=["csv"])
        trades_file = st.file_uploader("Upload trades CSV", type=["csv"])
    else:
        holdings_file = None
        trades_file = None
        holdings_path = st.text_input("Holdings CSV path", value=DEFAULT_HOLDINGS_PATH)
        trades_path = st.text_input("Trades CSV path", value=DEFAULT_TRADES_PATH)

    st.markdown("---")
    st.markdown("### Index & Embeddings")
    rebuild_index = st.button("(Re)build index now")
    use_cache = st.checkbox("Use cached index if available", value=True)

    st.markdown("---")
    st.markdown("### LLM (optional)")
    use_llm = st.checkbox("Enable LLM (Qwen 0.5B)", value=False)
    llm_name = st.text_input("LLM model name", value=DEFAULT_LLM_NAME)
    st.markdown("Enable only if you installed `transformers` and `torch` and have resources.")

    st.markdown("---")
    st.markdown("### Retrieval settings")
    top_k_ui = st.slider("Top K contexts", 1, 10, DEFAULT_TOP_K)
    sim_threshold_ui = st.slider("Similarity threshold", 0.0, 1.0, float(DEFAULT_SIM_THRESHOLD), step=0.01)

# Load CSVs
try:
    if use_upload and holdings_file is not None and trades_file is not None:
        holdings_df = pd.read_csv(holdings_file)
        trades_df = pd.read_csv(trades_file)
    else:
        holdings_df = safe_read_csv(holdings_path)
        trades_df = safe_read_csv(trades_path)
except Exception as e:
    st.error(f"Failed to load CSVs: {e}")
    st.stop()

st.success(f"Loaded holdings: {len(holdings_df)} rows; trades: {len(trades_df)} rows")

# Column detection
fund_col_name = detect_fund_column(holdings_df)
pl_col = detect_pl_column(holdings_df, trades_df)
st.write("Using fund column:", fund_col_name)
st.write("Using P&L column:", pl_col)

# Build documents + metadatas
documents = []
metadatas = []
for i, r in holdings_df.iterrows():
    documents.append(row_to_text(r, prefix="HOLDING"))
    metadatas.append({"source": "holdings", "row_index": int(i)})
for i, r in trades_df.iterrows():
    documents.append(row_to_text(r, prefix="TRADE"))
    metadatas.append({"source": "trades", "row_index": int(i)})

if len(documents) == 0:
    st.error("No documents created from CSVs.")
    st.stop()

# Load embedding model (cached)
with st.spinner("Loading embedding model..."):
    embed_model_obj = load_embed_model(EMBED_MODEL_NAME)

# Build or load FAISS index
with st.spinner("Building or loading FAISS index..."):
    index, embeddings = build_or_load_faiss_index(documents, embed_model_obj, rebuild=rebuild_index)

st.success("FAISS index ready. n_vectors = %d" % (index.ntotal if hasattr(index, "ntotal") else len(embeddings)))

# Load LLM if requested
llm_pipeline = None
if use_llm:
    with st.spinner("Loading LLM pipeline (may take time)..."):
        llm_pipeline = load_llm_pipeline(llm_name)
        if llm_pipeline is None:
            st.warning("LLM pipeline failed to load. Continuing without LLM.")
            use_llm = False

# Quick inspection
st.subheader("Quick inspection")
if fund_col_name and fund_col_name in holdings_df.columns:
    st.dataframe(holdings_df[fund_col_name].value_counts().head(10).rename_axis(fund_col_name).reset_index(name="count"))
else:
    st.write("Fund column not detected automatically. Check CSV column names.")

# Query UI
st.subheader("Ask the bot")
col1, col2 = st.columns([3, 1])
with col1:
    default_q = f"Total holdings for {holdings_df[fund_col_name].unique()[0]}" if (fund_col_name in holdings_df.columns and len(holdings_df[fund_col_name].unique())>0) else ""
    user_query = st.text_input("Your question", value=default_q)
    run_q = st.button("Ask")
with col2:
    st.write(f"Top K: {top_k_ui}")
    st.write(f"Sim threshold: {sim_threshold_ui:.2f}")

if run_q and user_query:
    t0 = time.time()
    answer, contexts = ask_bot(user_query, index, documents, metadatas, embed_model_obj,
                               holdings_df, trades_df, pl_col, fund_col_name,
                               llm_pipeline=llm_pipeline, use_llm=use_llm,
                               top_k=top_k_ui, sim_threshold=sim_threshold_ui)
    t1 = time.time()
    st.markdown("### Answer")
    st.code(answer)
    st.markdown(f"*(Response time: {t1 - t0:.2f}s)*")

    if contexts:
        st.markdown("### Top retrieved contexts (snippet, similarity, source)")
        rows = []
        for doc_text, sim_score, src in contexts[:top_k_ui]:
            rows.append({"similarity": float(sim_score), "source": src, "snippet": doc_text[:400]})
        st.table(pd.DataFrame(rows))

st.markdown("---")
st.markdown(
    "- Deterministic queries (counts, top holdings, yearly P&L) are answered instantly without the LLM.\n"
    "- LLM is optional. If disabled, the system returns short extracted snippets from the top context.\n"
    "- Embeddings and the FAISS index are cached to disk to avoid recomputation across runs."
)
