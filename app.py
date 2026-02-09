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
SAMPLE_DATA_PATH = BASE_DIR / "data" / "derived" / "kcc_merged_2024_2025_sample.csv"
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
    env_path = os.environ.get("KCC_DATA_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
    if SAMPLE_DATA_PATH.exists():
        return SAMPLE_DATA_PATH
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

def get_top_query_texts(lf, limit: int = 50):
    import polars as pl
    out = (
        lf.filter(pl.col("QueryText").is_not_null())
        .with_columns(pl.col("QueryText").cast(pl.Utf8).str.strip_chars())
        .filter(pl.col("QueryText") != "")
        .group_by("QueryText")
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
        .limit(limit)
        .collect(streaming=True)
    )
    return out.to_pandas()


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

is_sample = data_path.name.endswith("_sample.csv") or data_path.resolve() == SAMPLE_DATA_PATH.resolve()
lf = get_lazyframe(str(data_path), data_path.suffix.lower() == ".parquet")

with st.sidebar:
    st.header("Filters")
    state = st.multiselect("State", get_distinct(lf, "StateName"))
    district = st.multiselect("District", get_distinct(lf, "DistrictName"))
    crop = st.multiselect("Crop", get_distinct(lf, "Crop"))
    query_type = st.multiselect("Query Type", get_distinct(lf, "QueryType"))
    month = st.selectbox("Month", ["All"] + get_months(lf))
    search = st.text_input("Search text", placeholder="Search in query text or answer")

filter_exprs = build_filters(state, district, crop, query_type, month, search)
lf_filtered = lf.filter(filter_exprs) if filter_exprs else lf
total_rows = get_row_count(lf)
filtered_rows = get_row_count(lf_filtered)

st.subheader("Dataset Overview")
overview_left, overview_right = st.columns(2)
with overview_left:
    st.metric("Total rows in dataset", f"{total_rows:,}")
with overview_right:
    st.metric("Rows after filters", f"{filtered_rows:,}")
if is_sample:
    st.info("Using sample dataset. Upload or set KCC_DATA_PATH to use the full dataset.")

st.subheader("Geo Insights")

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

st.subheader("Top Segments")
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

st.subheader("Query Drilldown")
tab1, tab2 = st.tabs(["Top Query Texts", "Raw Samples"])
with tab1:
    st.caption("Query text grouped by (Top 50)")
    top_queries = get_top_query_texts(lf_filtered, limit=50)
    if top_queries.empty:
        st.info("No query text available for the current filters.")
    else:
        st.dataframe(top_queries, use_container_width=True, height=420)

with tab2:
    st.caption("Representative query/answer samples with context")
    samples = get_samples(lf_filtered)
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
