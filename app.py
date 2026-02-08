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
DATA_PATH = BASE_DIR / "data" / "kcc_merged_2024_2025.csv"
ENV_PATH = BASE_DIR / "config" / "kcc.env"
HF_DATASET_REPO = "D3m1-g0d/kcc-24-25"
HF_DATASET_FILE = "kcc_merged_2024_2025.csv"
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

st.title("KCC Unified Analytics Dashboard (2024-2025)")
st.caption("Kisan Call Centre transcripts (2024-2025) - interactive exploration and geographic insights")

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
    if DATA_PATH.exists():
        return DATA_PATH
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


@st.cache_data(show_spinner=True)
def load_data(path: Path):
    df = pd.read_csv(path)
    # Normalize column types for filtering
    if "Month" in df.columns:
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    return df


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


def render_percent_pie(data: pd.DataFrame, label_col: str = "label", value_col: str = "count"):
    if data.empty:
        st.info("No data available.")
        return
    df = data.copy()
    total = df[value_col].sum()
    if total <= 0:
        st.info("No data available.")
        return
    df["percent"] = df[value_col] / total
    df["percent_label"] = (df["percent"] * 100).round(1).astype(str) + "%"
    df["show_label"] = df["percent"] >= 0.06
    pie_colors = [
        "#ffc27a",
        "#ffb464",
        "#ffa34b",
        "#ff9333",
        "#ff8320",
        "#f47a1a",
        "#e66f14",
        "#d9650f",
        "#cc5b0a",
        "#bf5106",
    ]
    base = (
        alt.Chart(df)
        .encode(
            theta=alt.Theta(f"{value_col}:Q"),
            color=alt.Color(f"{label_col}:N", scale=alt.Scale(range=pie_colors), legend=None),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Label"),
                alt.Tooltip(f"{value_col}:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="Percent", format=".1%"),
            ],
        )
        .properties(height=320)
    )
    pie = base.mark_arc(innerRadius=30, outerRadius=120)
    labels = base.mark_text(radius=140, size=11, color="#f7f7f7").encode(
        text=alt.condition(alt.datum.show_label, "percent_label:N", alt.value(""))
    )
    chart = (
        (pie + labels)
        .configure_view(strokeOpacity=0)
        .properties(background=PLOT_BG)
    )
    st.altair_chart(chart, use_container_width=True)


def normalize_text_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        # Keep Unicode letters for "All scripts" mode; strip punctuation/symbols.
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"[_\d]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


STOPWORDS = {
    "the", "is", "am", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "at",
    "by", "from", "with", "and", "or", "as", "an", "a", "it", "this", "that", "these", "those", "about",
    "regarding", "related", "asked", "asking", "query", "queries", "information", "farmer", "farmers",
    "kisan", "samman", "nidhi", "pm", "yojana", "status", "beneficiary", "scheme", "details", "na", "none",
    "call", "centre", "center", "phone", "mobile", "number", "no", "ask", "asked", "told", "said", "want",
    "wants", "need", "needed", "please", "farmerquery", "farmerquerys", "farmerquerytype",
    "provide", "contact", "information", "not", "how", "giving", "new", "when", "next", "date", "helpline",
}


def filtered_tokens(texts: pd.Series) -> pd.Series:
    toks = normalize_text_series(texts).str.split()
    cleaned = toks.apply(
        lambda arr: [t for t in arr if t and t not in STOPWORDS and len(t) > 2]
    )
    return cleaned


def top_ngrams(texts: pd.Series, n: int = 2, top_k: int = 15) -> pd.DataFrame:
    tokens = filtered_tokens(texts)
    counts = {}
    for arr in tokens:
        if not arr or len(arr) < n:
            continue
        for i in range(len(arr) - n + 1):
            gram = " ".join(arr[i : i + n])
            if len(gram) < 4:
                continue
            # Drop grams composed of generic fillers that sneak through.
            parts = gram.split()
            if any(
                p in {
                    "call", "centre", "center", "number", "ask", "asked", "information",
                    "status", "details", "scheme", "regarding", "related",
                }
                for p in parts
            ):
                continue
            counts[gram] = counts.get(gram, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["phrase", "count"])
    out = (
        pd.Series(counts, name="count")
        .sort_values(ascending=False)
        .head(top_k)
        .rename_axis("phrase")
        .reset_index()
    )
    return out


data_path = resolve_data_path()
if not data_path:
    st.warning("Data file not found locally and no Hugging Face dataset is configured.")
    st.stop()

df = load_data(data_path)

with st.sidebar:
    st.header("Filters")
    state = st.multiselect("State", sorted(df["StateName"].dropna().unique().tolist()))
    district = st.multiselect("District", sorted(df["DistrictName"].dropna().unique().tolist()))
    crop = st.multiselect("Crop", sorted(df["Crop"].dropna().unique().tolist()))
    query_type = st.multiselect("Query Type", sorted(df["QueryType"].dropna().unique().tolist()))
    month = st.selectbox("Month", ["All"] + sorted(df["Month"].dropna().unique().tolist()))
    search = st.text_input("Search text", placeholder="Search in query text or answer")

filtered = df
if state:
    filtered = filtered[filtered["StateName"].isin(state)]
if district:
    filtered = filtered[filtered["DistrictName"].isin(district)]
if crop:
    filtered = filtered[filtered["Crop"].isin(crop)]
if query_type:
    filtered = filtered[filtered["QueryType"].isin(query_type)]
if month != "All":
    filtered = filtered[filtered["Month"] == month]
if search:
    s = search.strip().lower()
    qt = filtered["QueryText"].fillna("").str.lower()
    ka = filtered["KccAns"].fillna("").str.lower()
    filtered = filtered[qt.str.contains(s) | ka.str.contains(s)]

st.subheader("Geo Insights")

def norm(s):
    return str(s).strip().lower().replace(" ", "")

map_left, map_right = st.columns(2)

with map_left:
    st.caption("District Query Volume")
    district_geojson_path = Path("/home/sneharup/KCC/apt/data/derived/india_districts.geojson")
    if district_geojson_path.exists():
        district_geojson = json.loads(district_geojson_path.read_text(encoding="utf-8"))
        district_counts = (
            filtered.groupby(["StateName", "DistrictName"], dropna=False)
            .size()
            .reset_index(name="count")
        )
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
        st.info("Add /home/sneharup/KCC/apt/data/derived/india_districts.geojson to enable district map.")

with map_right:
    st.caption("State Query Volume")
    geojson_path = Path("/home/sneharup/KCC/apt/data/derived/india_states.geojson")
    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        state_counts = filtered["StateName"].value_counts().reset_index()
        state_counts.columns = ["StateName", "count"]
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
        st.info("Add /home/sneharup/KCC/apt/data/derived/india_states.geojson to enable the state map.")

st.subheader("Top Segments")
row1_a, row1_b = st.columns(2)
with row1_a:
    st.write("Top Districts (Top 10)")
    top_districts = (
        filtered["DistrictName"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
        .dropna()
        .value_counts()
        .head(10)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    render_ordered_bar(top_districts)
with row1_b:
    st.write("Top States (Top 10)")
    top_states = (
        filtered["StateName"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
        .dropna()
        .value_counts()
        .head(10)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    render_ordered_bar(top_states)

row2_a, row2_b = st.columns(2)
with row2_a:
    st.write("Top Query Types (Top 10)")
    clean_qt = filtered["QueryType"].astype(str).str.replace("\t", "", regex=False).str.strip()
    top_query_types = (
        clean_qt.value_counts()
        .head(10)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    render_ordered_bar(top_query_types)
with row2_b:
    st.write("Top Crops (Top 10)")
    top_crops = (
        filtered["Crop"]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(10)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    render_ordered_bar(top_crops)

st.subheader("Top Segments (Share %)")
pie1_a, pie1_b = st.columns(2)
with pie1_a:
    st.write("Top Districts (Top 10)")
    render_percent_pie(top_districts)
with pie1_b:
    st.write("Top States (Top 10)")
    render_percent_pie(top_states)

pie2_a, pie2_b = st.columns(2)
with pie2_a:
    st.write("Top Query Types (Top 10)")
    render_percent_pie(top_query_types)
with pie2_b:
    st.write("Top Crops (Top 10)")
    render_percent_pie(top_crops)

st.subheader("Query Drilldown")
drill = filtered.copy()

tab1, tab2 = st.tabs(["Most Frequent Phrases", "Raw Samples"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top Bigrams")
        bi = top_ngrams(
            drill["QueryText"] if len(drill) else pd.Series(dtype=str),
            n=2,
            top_k=12,
        )
        if len(bi):
            render_ordered_bar(bi.rename(columns={"phrase": "label"}), label_col="label", value_col="count")
        else:
            st.info("No phrase data available.")
    with c2:
        st.caption("Bigram Cloud (cleaned)")
        bigram_cloud = top_ngrams(
            drill["QueryText"] if len(drill) else pd.Series(dtype=str),
            n=2,
            top_k=35,
        )
        if len(bigram_cloud):
            max_c = max(bigram_cloud["count"].max(), 1)
            html = []
            for _, r in bigram_cloud.iterrows():
                size = 12 + int((r["count"] / max_c) * 22)
                opacity = 0.55 + (r["count"] / max_c) * 0.45
                html.append(
                    f"<span style='font-size:{size}px; opacity:{opacity:.2f}; margin:6px 8px; display:inline-block; color:{ACCENT};'>{r['phrase']}</span>"
                )
            st.markdown(
                f"""<div style="background:{PLOT_BG}; border:1px solid #2a2a2a; border-radius:10px; padding:12px; line-height:1.8; height:320px; overflow:auto;">{''.join(html)}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No keyword data available.")

with tab2:
    st.caption("Representative query/answer samples with context")
    sample_cols = ["StateName", "DistrictName", "Crop", "QueryType", "Month", "QueryText", "KccAns"]
    samples = drill[sample_cols].dropna(subset=["QueryText"]).copy()
    samples["ans_len"] = samples["KccAns"].astype(str).str.strip().str.len()
    samples = samples.sort_values("ans_len", ascending=False).head(20)
    if samples.empty:
        st.info("No sample rows available for this query type.")
    else:
        for i, row in samples.iterrows():
            meta = (
                f"{row.get('StateName', 'NA')} | {row.get('DistrictName', 'NA')} | "
                f"Crop: {row.get('Crop', 'NA')} | QueryType: {row.get('QueryType', 'NA')} | "
                f"Month: {row.get('Month', 'NA')}"
            )
            st.markdown(f"<span style='color:{ACCENT_SOFT}; font-weight:700;'>{meta}</span>", unsafe_allow_html=True)
            st.markdown(f"- Query: {str(row.get('QueryText', '')).strip()[:400]}")
            ans = str(row.get("KccAns", "")).strip()
            if ans:
                st.markdown(f"- Answer: {ans[:400]}")
            st.markdown("---")
