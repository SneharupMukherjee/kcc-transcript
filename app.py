from pathlib import Path

import json
import os

import pandas as pd
import streamlit as st
import altair as alt
from bokeh.models import ColorBar, GeoJSONDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="KCC Unified Explorer (2024-2025)", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / "config" / "kcc.env"
HF_DATASET_REPO = "D3m1-g0d/kcc-24-25-up"
HF_DATASET_FILE = "kcc_merged_2024_2025_up.parquet"
APP_BG = "#050505"
PLOT_BG = "#050505"
ACCENT = "#ff9f43"
ACCENT_SOFT = "#ffc27a"
MAP_ORANGE_PALETTE = [
    "#2b1a0d",
    "#4a280e",
    "#673512",
    "#854216",
    "#a54f1a",
    "#c4601f",
    "#dd7b2f",
    "#ee9a58",
    "#ffc27a",
]

st.markdown(
    """
    <style>
    :root {
      --bg: #050505;
      --panel: #111111;
      --panel-2: #1a1a1a;
      --text: #f7f7f7;
      --muted: #bdbdbd;
      --accent: #ff8c00;
      --accent-2: #ffb347;
      --border: #2a2a2a;
    }
    .stApp {
      background: radial-gradient(circle at 10% 0%, #1b1206 0%, var(--bg) 35%);
      color: var(--text);
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
      color: var(--text);
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0a0a0a 0%, #130d06 100%);
      border-right: 1px solid var(--border);
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    .stSlider {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .stButton button, .stDownloadButton button {
      background: linear-gradient(135deg, var(--accent), #d96f00);
      color: #111 !important;
      border: none;
      font-weight: 700;
    }
    .stButton button:hover, .stDownloadButton button:hover {
      background: linear-gradient(135deg, var(--accent-2), var(--accent));
    }
    [data-testid="stMetricValue"] {
      color: var(--accent);
    }
    [data-testid="stMetricLabel"] {
      color: var(--muted);
    }
    [data-testid="stDataFrame"], .stAlert {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("KCC Uttar Pradesh Analytics Dashboard (2024-2025)")
st.caption("Kisan Call Centre transcripts for Uttar Pradesh (2024-2025) - interactive exploration and geographic insights")

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file(ENV_PATH)


@st.cache_resource(show_spinner=True)
def resolve_data_path() -> Path | None:
    repo_id = os.environ.get("HF_DATASET_REPO", HF_DATASET_REPO)
    filename = os.environ.get("HF_DATASET_FILE", HF_DATASET_FILE)
    token = os.environ.get("HF_TOKEN") or None
    if not repo_id or not filename:
        return None
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        st.error(f"Failed to download dataset from Hugging Face: {exc}")
        return None
    return Path(downloaded)


@st.cache_resource(show_spinner=True)
def get_lazyframe(path: str, is_parquet: bool):
    try:
        import polars as pl  # local import to avoid hard crash during app boot
    except Exception as exc:
        st.error(f"Polars failed to import: {exc}")
        st.stop()
    if is_parquet:
        return pl.scan_parquet(path)
    return pl.scan_csv(path, ignore_errors=True)


def build_filters(state, district, crop, query_type, month, search):
    import polars as pl
    exprs = []
    if state:
        exprs.append(pl.col("StateName").is_in(state))
    if district:
        exprs.append(pl.col("DistrictName").is_in(district))
    if crop:
        exprs.append(pl.col("Crop").is_in(crop))
    if query_type:
        exprs.append(pl.col("QueryType").is_in(query_type))
    if month != "All":
        exprs.append(pl.col("Month").cast(pl.Int64, strict=False) == month)
    if search:
        s = search.strip().lower()
        exprs.append(
            pl.col("QueryText").fill_null("").str.to_lowercase().str.contains(s)
            | pl.col("KccAns").fill_null("").str.to_lowercase().str.contains(s)
        )
    return exprs


def get_distinct(lf, column: str):
    import polars as pl
    out = (
        lf.select(pl.col(column).cast(pl.Utf8).str.strip_chars())
        .filter(pl.col(column).is_not_null() & (pl.col(column) != ""))
        .unique()
        .sort(column)
        .collect(streaming=True)
    )
    return out[column].to_list()


def get_months(lf):
    import polars as pl
    out = (
        lf.select(pl.col("Month").cast(pl.Int64, strict=False).alias("m"))
        .drop_nulls()
        .unique()
        .sort("m")
        .collect(streaming=True)
    )
    return out["m"].to_list()


def get_row_count(lf):
    import polars as pl
    return lf.select(pl.len()).collect(streaming=True).item()


def get_counts(lf, group_cols: list[str], limit: int | None = None):
    import polars as pl
    out = lf.group_by(group_cols).len().rename({"len": "count"}).sort("count", descending=True)
    if limit:
        out = out.limit(limit)
    return out.collect(streaming=True).to_pandas()


def get_samples(lf):
    import polars as pl
    out = (
        lf.select(
            "StateName",
            "DistrictName",
            "Crop",
            "QueryType",
            "Month",
            "QueryText",
            "KccAns",
            pl.col("KccAns").fill_null("").cast(pl.Utf8).str.strip_chars().str.len_bytes().alias("ans_len"),
        )
        .filter(pl.col("QueryText").is_not_null())
        .sort("ans_len", descending=True)
        .limit(20)
        .collect(streaming=True)
    )
    return out.to_pandas()


def normalize_query_text(col):
    import polars as pl
    expr = (
        col.cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\bpm\s+ksny\b", " ")
        .str.replace_all(r"\b(the|on|a|of|information|about|to|in|asked|me|tell|beneficiary|beneficiaries|please|give|checking|and|under|how|for|provide|know|knowing|your|regarding|ask|related|while)\b", " ")
        .str.replace_all(r"\bpradhan\s+mantri\b", "pm")
        .str.replace_all(r"\bprime\s+minister\b", "pm")
        .str.replace_all(r"\byojana\b", "scheme")
        .str.replace_all(r"\byojna\b", "scheme")
        .str.replace_all(r"pm\s+kisan\s+samman\s+nidhi", "pmksn")
        .str.replace_all(r"\bpm\s+kisan\b", "pmksn")
        .str.replace_all(r"kisan\s+samman\s+nidhi", "pmksn")
    )
    expr = pl.when(expr.str.contains(r"\bstatus\b")).then(
        pl.concat_str(
            [
                pl.lit("status "),
                expr.str.replace_all(r"\bstatus\b", " "),
            ]
        )
    ).otherwise(expr)
    expr = expr.str.replace_all(r"\bscheme\b", " ")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()


def get_group_samples(lf, group_text: str, limit: int = 25):
    import polars as pl
    cleaned = normalize_query_text(pl.col("QueryText"))
    out = (
        lf.filter(pl.col("QueryText").is_not_null())
        .with_columns(cleaned.alias("QueryTextNorm"))
        .filter(pl.col("QueryTextNorm") == group_text)
        .select("QueryText", "KccAns")
        .limit(limit)
        .collect(streaming=True)
    )
    return out.to_pandas()

def get_top_descriptive_answers(lf, limit: int | None = None):
    import polars as pl
    cleaned = normalize_query_text(pl.col("QueryText"))
    out = (
        lf.filter(pl.col("QueryText").is_not_null())
        .with_columns(
            cleaned.alias("QueryTextNorm"),
            pl.col("KccAns")
            .fill_null("")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.len_bytes()
            .alias("AnsLen"),
        )
        .filter(pl.col("QueryTextNorm") != "")
        .group_by("QueryTextNorm")
        .agg(
            pl.len().alias("count"),
            pl.max("AnsLen").alias("max_ans_len"),
            pl.mean("AnsLen").alias("avg_ans_len"),
        )
        .sort("max_ans_len", descending=True)
    )
    if limit is not None:
        out = out.limit(limit)
    out = out.collect(streaming=True)
    return out.rename({"QueryTextNorm": "QueryText"}).to_pandas()

def get_top_query_texts(lf, limit: int | None = None):
    import polars as pl
    cleaned = normalize_query_text(pl.col("QueryText"))
    out = (
        lf.filter(pl.col("QueryText").is_not_null())
        .with_columns(cleaned.alias("QueryTextNorm"))
        .filter(pl.col("QueryTextNorm") != "")
        .group_by("QueryTextNorm")
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
    )
    if limit is not None:
        out = out.limit(limit)
    out = out.collect(streaming=True)
    return out.rename({"QueryTextNorm": "QueryText"}).to_pandas()

@st.cache_data(show_spinner=False)
def load_intent_summaries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def render_ordered_bar(data: pd.DataFrame, label_col: str = "label", value_col: str = "count"):
    chart = (
        alt.Chart(data)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(f"{label_col}:N", sort="-y", title=None),
            y=alt.Y(f"{value_col}:Q"),
            tooltip=[label_col, value_col],
        )
        .properties(height=320)
        .properties(background=PLOT_BG)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#f7f7f7",
            titleColor="#f7f7f7",
            gridColor="#2a2a2a",
            domainColor="#2a2a2a",
            tickColor="#2a2a2a",
        )
    )
    st.altair_chart(chart, use_container_width=True)



data_path = resolve_data_path()
if not data_path:
    st.warning("Data file not found locally and no Hugging Face dataset is configured.")
    st.stop()

lf = get_lazyframe(str(data_path), data_path.suffix.lower() == ".parquet")

with st.sidebar:
    st.header("Filters (Uttar Pradesh only)")
    district = st.multiselect("District", get_distinct(lf, "DistrictName"))
    crop = st.multiselect("Crop", get_distinct(lf, "Crop"))
    query_type_options = get_distinct(lf, "QueryType")
    default_query_type = []
    for opt in query_type_options:
        if "government" in str(opt).lower() and "scheme" in str(opt).lower():
            default_query_type = [opt]
            break
    query_type = st.multiselect("Query Type", query_type_options, default=default_query_type)
    month = st.selectbox("Month", ["All"] + get_months(lf))
    search = st.text_input("Search text", placeholder="Search in query text or answer")

filter_exprs = build_filters([], district, crop, query_type, month, search)
lf_filtered = lf.filter(filter_exprs) if filter_exprs else lf
total_rows = get_row_count(lf)
filtered_rows = get_row_count(lf_filtered)

st.subheader("Dataset Overview (Uttar Pradesh)")
overview_left, overview_right = st.columns(2)
with overview_left:
    st.metric("Total rows in dataset", f"{total_rows:,}")
with overview_right:
    st.metric("Rows after filters", f"{filtered_rows:,}")

st.subheader("Geo Insights (Uttar Pradesh)")

def norm(s):
    return str(s).strip().lower().replace(" ", "")

map_left, map_right = st.columns(2)

with map_left:
    st.caption("District Query Volume")
    district_geojson_path = BASE_DIR / "data" / "derived" / "india_districts.geojson"
    if district_geojson_path.exists():
        district_geojson = json.loads(district_geojson_path.read_text(encoding="utf-8"))
        district_counts = get_counts(lf_filtered, ["StateName", "DistrictName"])
        district_counts["StateKey"] = district_counts["StateName"].apply(norm)
        district_counts["DistrictKey"] = district_counts["DistrictName"].apply(norm)
        district_lookup = {
            (r["StateKey"], r["DistrictKey"]): int(r["count"])
            for _, r in district_counts.iterrows()
        }
        for feat in district_geojson.get("features", []):
            props = feat.get("properties", {})
            state_name = props.get("StateName") or props.get("STATE_NAME") or props.get("state") or props.get("NAME_1")
            district_name = props.get("DistrictName") or props.get("DISTRICT") or props.get("district") or props.get("NAME_2")
            props["StateName"] = state_name
            props["DistrictName"] = district_name
            props["count"] = district_lookup.get((norm(state_name), norm(district_name)), 0)
        palette = MAP_ORANGE_PALETTE
        d_counts = [f["properties"].get("count", 0) for f in district_geojson.get("features", [])]
        color_mapper = LinearColorMapper(palette=palette, low=min(d_counts) if d_counts else 0, high=max(d_counts) if d_counts else 1, nan_color="#151515")
        geosource = GeoJSONDataSource(geojson=json.dumps(district_geojson))
        p2 = figure(title="Queries by District", height=760, width=760, toolbar_location="below", tools="pan,wheel_zoom,box_zoom,reset,save")
        p2.background_fill_color = PLOT_BG
        p2.border_fill_color = PLOT_BG
        p2.outline_line_color = "#2a2a2a"
        p2.xgrid.grid_line_color = "#2a2a2a"
        p2.ygrid.grid_line_color = "#2a2a2a"
        p2.xaxis.major_label_text_color = "#f7f7f7"
        p2.yaxis.major_label_text_color = "#f7f7f7"
        p2.title.text_color = ACCENT_SOFT
        renderer2 = p2.patches("xs", "ys", source=geosource, fill_color={"field": "count", "transform": color_mapper}, line_color="grey", line_width=0.15, fill_alpha=1)
        hover2 = HoverTool(renderers=[renderer2])
        hover2.tooltips = """
            <div style="background:#121212; border:1px solid #2a2a2a; border-radius:8px; padding:8px 10px; color:#f7f7f7;">
                <div><span style="color:#ffc27a;">State:</span> @StateName</div>
                <div><span style="color:#ffc27a;">District:</span> @DistrictName</div>
                <div><span style="color:#ffc27a;">Queries:</span> @count{0,0}</div>
            </div>
        """
        p2.add_tools(hover2)
        cb2 = ColorBar(color_mapper=color_mapper, width=8)
        cb2.background_fill_color = PLOT_BG
        cb2.background_fill_alpha = 1.0
        cb2.border_line_color = "#2a2a2a"
        cb2.major_label_text_color = "#f7f7f7"
        cb2.title_text_color = ACCENT_SOFT
        p2.add_layout(cb2, "right")
        st.bokeh_chart(p2, use_container_width=True)
    else:
        st.info("Add data/derived/india_districts.geojson to enable district map.")

with map_right:
    st.caption("State Query Volume")
    geojson_path = BASE_DIR / "data" / "derived" / "india_states.geojson"
    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        state_counts = get_counts(lf_filtered, ["StateName"])
        state_counts["StateKey"] = state_counts["StateName"].apply(norm)
        state_lookup = state_counts.set_index("StateKey")["count"]
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            name = props.get("StateName") or props.get("STATE_NAME") or props.get("state")
            props["StateName"] = name
            props["count"] = int(state_lookup.get(norm(name), 0))
        palette = MAP_ORANGE_PALETTE
        counts = [f["properties"].get("count", 0) for f in geojson.get("features", [])]
        color_mapper = LinearColorMapper(palette=palette, low=min(counts) if counts else 0, high=max(counts) if counts else 1, nan_color="#151515")
        geosource = GeoJSONDataSource(geojson=json.dumps(geojson))
        p = figure(title="Queries by State", height=760, width=760, toolbar_location="below", tools="pan,wheel_zoom,box_zoom,reset,save")
        p.background_fill_color = PLOT_BG
        p.border_fill_color = PLOT_BG
        p.outline_line_color = "#2a2a2a"
        p.xgrid.grid_line_color = "#2a2a2a"
        p.ygrid.grid_line_color = "#2a2a2a"
        p.xaxis.major_label_text_color = "#f7f7f7"
        p.yaxis.major_label_text_color = "#f7f7f7"
        p.title.text_color = ACCENT_SOFT
        renderer = p.patches("xs", "ys", source=geosource, fill_color={"field": "count", "transform": color_mapper}, line_color="grey", line_width=0.3, fill_alpha=1)
        hover = HoverTool(renderers=[renderer])
        hover.tooltips = """
            <div style="background:#121212; border:1px solid #2a2a2a; border-radius:8px; padding:8px 10px; color:#f7f7f7;">
                <div><span style="color:#ffc27a;">State:</span> @StateName</div>
                <div><span style="color:#ffc27a;">Queries:</span> @count{0,0}</div>
            </div>
        """
        p.add_tools(hover)
        cb = ColorBar(color_mapper=color_mapper, width=8)
        cb.background_fill_color = PLOT_BG
        cb.background_fill_alpha = 1.0
        cb.border_line_color = "#2a2a2a"
        cb.major_label_text_color = "#f7f7f7"
        cb.title_text_color = ACCENT_SOFT
        p.add_layout(cb, "right")
        st.bokeh_chart(p, use_container_width=True)
    else:
        st.info("Add data/derived/india_states.geojson to enable the state map.")

st.subheader("Top Segments (Uttar Pradesh)")
row1_a, row1_b = st.columns(2)
with row1_a:
    st.write("Top Districts (Top 10)")
    top_districts = get_counts(lf_filtered, ["DistrictName"], limit=10)
    top_districts = top_districts.rename(columns={"DistrictName": "label"}).sort_values("count", ascending=False)
    render_ordered_bar(top_districts)
with row1_b:
    st.write("Top States (Top 10)")
    top_states = get_counts(lf_filtered, ["StateName"], limit=10)
    top_states = top_states.rename(columns={"StateName": "label"}).sort_values("count", ascending=False)
    render_ordered_bar(top_states)

row2_a, row2_b = st.columns(2)
with row2_a:
    st.write("Top Query Types (Top 10)")
    top_query_types = get_counts(lf_filtered, ["QueryType"], limit=10)
    top_query_types = top_query_types.rename(columns={"QueryType": "label"}).sort_values("count", ascending=False)
    render_ordered_bar(top_query_types)
with row2_b:
    st.write("Top Crops (Top 10)")
    top_crops = get_counts(lf_filtered, ["Crop"], limit=10)
    top_crops = top_crops.rename(columns={"Crop": "label"}).sort_values("count", ascending=False)
    render_ordered_bar(top_crops)

st.subheader("Query Drilldown (Uttar Pradesh)")
st.caption("Top normalized query texts (All)")
left, right = st.columns([1.2, 1])
with left:
    top_queries = get_top_query_texts(lf_filtered, limit=None)
    if top_queries.empty:
        st.info("No query text available for the current filters.")
        selected_text = None
    else:
        total_rows = len(top_queries)
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            page_size = st.selectbox("Rows per page", [25, 50, 100, 200, 500], index=1, key="topq_page_size")
        total_pages = max((total_rows - 1) // page_size + 1, 1)
        with nav_right:
            page = st.selectbox("Page", list(range(1, total_pages + 1)), index=0, key="topq_page")
        start = (page - 1) * page_size
        end = min(start + page_size, total_rows)
        page_df = top_queries.iloc[start:end].reset_index(drop=True)
        table_key = "top_queries_table"
        st.session_state["topq_page_df"] = page_df
        st.session_state["topq_page_idx"] = page

        def _handle_topq_select():
            table_state = st.session_state.get(table_key, {})
            rows = table_state.get("selection", {}).get("rows", [])
            selected = None
            if rows:
                idx = rows[0]
                df = st.session_state.get("topq_page_df")
                if df is not None and 0 <= idx < len(df):
                    selected = str(df.iloc[idx]["QueryText"])
            st.session_state["topq_selected_text"] = selected

        st.dataframe(
            page_df,
            use_container_width=True,
            height=520,
            on_select=_handle_topq_select,
            selection_mode="single-row",
            key=table_key,
        )
        if st.session_state.get("topq_page_idx") != page:
            st.session_state["topq_selected_text"] = None
        selected_text = st.session_state.get("topq_selected_text")

with right:
    st.caption("Query + Answer samples (selected group)")
    if not selected_text:
        st.info("Select a row on the left to see samples.")
    else:
        with st.spinner("Loading samples..."):
            samples = get_group_samples(lf_filtered, str(selected_text), limit=25)
        if samples.empty:
            st.info("No samples found for this group.")
        else:
            for _, row in samples.iterrows():
                q = str(row.get("QueryText", "")).strip()
                a = str(row.get("KccAns", "")).strip()
                st.markdown(f"- Query: {q[:400]}")
                if a:
                    st.markdown(f"- Answer: {a[:400]}")
                st.markdown("---")

st.subheader("Descriptive Answers (Uttar Pradesh)")
st.caption("Intent clusters with representative/descriptive questions and best answers (precomputed)")

intent_path = BASE_DIR / "data" / "derived" / "kcc_intent_summaries.parquet"
intent_df = load_intent_summaries(intent_path)
if intent_df.empty:
    st.info("Run scripts/build_intent_clusters.py to generate intent summaries.")
else:
    left_i, right_i = st.columns([1.2, 1])
    with left_i:
        table_key = "intent_summary_table"
        table = st.dataframe(
            intent_df[["question_cluster_id", "rep_question", "desc_question"]],
            use_container_width=True,
            height=520,
            on_select="rerun",
            selection_mode="single-row",
            key=table_key,
        )
        rows = getattr(table, "selection", None).rows if table is not None else []
        selected_row = rows[0] if rows else None
    with right_i:
        if selected_row is None:
            st.info("Select an intent row to see best answers.")
        else:
            row = intent_df.iloc[selected_row]
            st.markdown(f"**Representative Q:** {row.get('rep_question', '')}")
            st.markdown(f"**Descriptive Q:** {row.get('desc_question', '')}")
            st.markdown("---")
            st.markdown("**Best Answer (overall):**")
            st.markdown(str(row.get("best_answer_overall", ""))[:800])
            st.markdown("---")
            st.markdown("**Best Answers (per answer cluster):**")
            try:
                per_cluster = json.loads(row.get("best_answer_per_answer_cluster", "[]"))
            except Exception:
                per_cluster = []
            if not per_cluster:
                st.info("No per-cluster answers available.")
            else:
                for item in per_cluster:
                    st.markdown(f"- Cluster {item.get('answer_cluster_id')}: {str(item.get('KccAns', ''))[:800]}")
