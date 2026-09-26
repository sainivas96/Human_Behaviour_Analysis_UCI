import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Human Behaviour Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_har.csv"

# Official UCI HAR download.
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)

ACTIVITY_MAP = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data(show_spinner=True)
def load_uci():
    """Download and load the official UCI HAR dataset.

    UCI currently serves an outer ZIP containing the actual
    ``UCI HAR Dataset.zip`` archive, so this loader supports both the
    nested and direct layouts.
    """
    try:
        request = urllib.request.Request(
            UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 Human-Behaviour-Analysis"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            zip_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the official UCI HAR dataset. "
            "Check the internet connection and try again."
        ) from exc

    try:
        # Open the downloaded archive. If it contains another ZIP, use the
        # inner archive because that is where features.txt and train/test live.
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
            outer_names = outer_zip.namelist()
            inner_zip_name = next(
                (
                    n for n in outer_names
                    if n.replace("\\", "/").lower().endswith("uci har dataset.zip")
                ),
                None,
            )

            if inner_zip_name:
                dataset_bytes = outer_zip.read(inner_zip_name)
                z = zipfile.ZipFile(io.BytesIO(dataset_bytes))
            else:
                z = outer_zip

            with z:
                names = z.namelist()

                def find_member(filename):
                    target = filename.replace("\\", "/").lower()
                    matches = [
                        n for n in names
                        if n.replace("\\", "/").lower().endswith(target)
                    ]
                    if not matches:
                        raise FileNotFoundError(f"UCI file not found: {filename}")
                    return matches[0]

                feature_file = find_member("features.txt")
                x_train_file = find_member("train/X_train.txt")
                y_train_file = find_member("train/y_train.txt")
                subject_train_file = find_member("train/subject_train.txt")
                x_test_file = find_member("test/X_test.txt")
                y_test_file = find_member("test/y_test.txt")
                subject_test_file = find_member("test/subject_test.txt")

                feature_lines = (
                    z.read(feature_file)
                    .decode("utf-8", errors="replace")
                    .splitlines()
                )
                feature_names = []
                for line in feature_lines:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        feature_names.append(parts[1].strip())

                if len(feature_names) != 561:
                    raise ValueError(
                        f"Expected 561 feature names, found {len(feature_names)}."
                    )

                seen = {}
                unique_feature_names = []
                for name in feature_names:
                    count = seen.get(name, 0)
                    unique_feature_names.append(
                        name if count == 0 else f"{name}_{count}"
                    )
                    seen[name] = count + 1

                def read_matrix(member):
                    return np.loadtxt(
                        io.BytesIO(z.read(member)), dtype=np.float64
                    )

                def read_vector(member):
                    return np.loadtxt(
                        io.BytesIO(z.read(member)), dtype=np.int64
                    )

                x_train = read_matrix(x_train_file)
                x_test = read_matrix(x_test_file)
                y_train = read_vector(y_train_file)
                y_test = read_vector(y_test_file)
                subject_train = read_vector(subject_train_file)
                subject_test = read_vector(subject_test_file)

                if x_train.ndim != 2 or x_train.shape[1] != 561:
                    raise ValueError(
                        f"Unexpected X_train shape: {x_train.shape}; expected (*, 561)."
                    )
                if x_test.ndim != 2 or x_test.shape[1] != 561:
                    raise ValueError(
                        f"Unexpected X_test shape: {x_test.shape}; expected (*, 561)."
                    )

                if not (
                    len(x_train) == len(y_train) == len(subject_train)
                    and len(x_test) == len(y_test) == len(subject_test)
                ):
                    raise ValueError("UCI train/test files have mismatched row counts.")

                train = pd.DataFrame(x_train, columns=unique_feature_names)
                train["activity"] = y_train
                train["subject"] = subject_train

                test = pd.DataFrame(x_test, columns=unique_feature_names)
                test["activity"] = y_test
                test["subject"] = subject_test

                df = pd.concat([train, test], ignore_index=True)
                df["activity"] = df["activity"].map(ACTIVITY_MAP)

                if df["activity"].isna().any():
                    raise ValueError("Unexpected activity labels found in UCI dataset.")

                return df

    except Exception as exc:
        raise RuntimeError(
            f"The UCI HAR ZIP was downloaded, but its files could not be parsed: {exc}"
        ) from exc


def get_sensor_columns(df):
    preferred = [
        "body_accel_magnitude",
        "gyro_magnitude",
        "dominant_frequency_hz",
    ]
    present = [c for c in preferred if c in df.columns]
    if present:
        return present

    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["mean", "std", "max", "freq"])
    ]
    return candidates[:12]


st.markdown(
    """
    <style>
    .main-title {font-size: 2.35rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1rem; color: #667085; margin-bottom: 1.5rem;}
    .insight {padding: 1rem 1.1rem; border: 1px solid #e4e7ec; border-radius: 14px;
              background: #fafafa; margin-bottom: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 Human Behaviour Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive analysis of smartphone sensor data to discover '
    'human activity patterns, movement intensity, and behavioural trends.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose dataset",
        ["Included demo dataset", "Official UCI HAR dataset"],
        index=0,
    )
    st.caption("The demo dataset is included so the app runs immediately.")
    st.divider()
    st.header("Filters")

try:
    if source == "Official UCI HAR dataset":
        df = load_uci()
        st.success("Official UCI HAR dataset loaded successfully.")
    else:
        df = load_sample()
except Exception as exc:
    st.error("The official dataset could not be loaded. Showing the included demo dataset.")
    st.caption(f"Details: {exc}")
    df = load_sample()

# Clean labels and basic preprocessing.
df = df.copy()
df.columns = [str(c).strip() for c in df.columns]
if "activity" in df.columns:
    df["activity"] = df["activity"].astype(str).str.replace("_", " ").str.title()

if "subject" in df.columns:
    subjects = sorted(df["subject"].dropna().unique().tolist())
    chosen_subjects = st.sidebar.multiselect(
        "Subjects",
        subjects,
        default=subjects[:min(10, len(subjects))],
    )
    if chosen_subjects:
        df = df[df["subject"].isin(chosen_subjects)]

activities_available = sorted(df["activity"].dropna().unique()) if "activity" in df.columns else []
chosen_activities = st.sidebar.multiselect(
    "Activities",
    activities_available,
    default=activities_available,
)
if chosen_activities and "activity" in df.columns:
    df = df[df["activity"].isin(chosen_activities)]

st.sidebar.caption(f"Rows after filters: {len(df):,}")

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
feature_count = len([c for c in df.columns if c not in {"activity", "subject"}])
c2.metric("Features", f"{feature_count:,}")
c3.metric("Activities", f"{df['activity'].nunique() if 'activity' in df else 0:,}")
c4.metric("Subjects", f"{df['subject'].nunique() if 'subject' in df else 0:,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Behaviour Patterns", "Feature Analysis", "Data Explorer"]
)

with tab1:
    st.subheader("Activity distribution")
    if "activity" in df.columns:
        counts = df["activity"].value_counts().rename_axis("activity").reset_index(name="count")
        fig = px.bar(
            counts,
            x="activity",
            y="count",
            text="count",
            title="Observations by activity",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Observations")
        st.plotly_chart(fig, width="stretch")

        top_activity = counts.iloc[0]["activity"]
        st.markdown(
            f'<div class="insight"><b>Key insight:</b> {top_activity} has the highest '
            f'number of observations in the current filtered dataset.</div>',
            unsafe_allow_html=True,
        )

    if "subject" in df.columns and "activity" in df.columns:
        subject_activity = (
            df.groupby(["subject", "activity"])
            .size()
            .reset_index(name="observations")
        )
        fig2 = px.density_heatmap(
            subject_activity,
            x="activity",
            y="subject",
            z="observations",
            histfunc="sum",
            title="Subject × activity coverage",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, width="stretch")

with tab2:
    st.subheader("Behaviour patterns")
    sensor_cols = get_sensor_columns(df)

    if sensor_cols:
        metric = st.selectbox("Select a sensor-derived feature", sensor_cols)
        if "activity" in df.columns:
            fig = px.box(
                df,
                x="activity",
                y=metric,
                points=False,
                title=f"{metric} across activities",
            )
            fig.update_layout(xaxis_title="", yaxis_title=metric)
            st.plotly_chart(fig, width="stretch")

            group_stats = (
                df.groupby("activity")[metric]
                .agg(["mean", "median", "std"])
                .round(4)
                .sort_values("mean", ascending=False)
            )
            st.dataframe(group_stats, width="stretch")

    if "subject" in df.columns and "activity" in df.columns:
        st.subheader("Activity mix by subject")
        mix = (
            df.groupby(["subject", "activity"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            mix,
            x="subject",
            y="count",
            color="activity",
            barmode="stack",
            title="How activity observations are distributed across subjects",
        )
        st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Feature analysis")
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != "subject"
    ]

    if len(numeric_cols) >= 2:
        corr_cols = numeric_cols[: min(15, len(numeric_cols))]
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=False,
            aspect="auto",
            title="Correlation matrix for representative numeric features",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig, width="stretch")

    sensor_cols = get_sensor_columns(df)
    if sensor_cols:
        selected = st.multiselect(
            "Select features for distribution comparison",
            sensor_cols,
            default=sensor_cols[: min(3, len(sensor_cols))],
        )
        if selected:
            long_df = df[selected].melt(var_name="feature", value_name="value")
            fig = px.histogram(
                long_df,
                x="value",
                facet_row="feature",
                title="Feature distributions",
                nbins=40,
            )
            fig.update_layout(height=max(450, 220 * len(selected)))
            st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Filtered dataset")
    st.dataframe(df.head(500), width="stretch", height=420)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name="human_behaviour_analysis_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Data quality summary")
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, width="stretch")

st.divider()
st.caption(
    "Dataset reference: Reyes-Ortiz et al., Human Activity Recognition Using Smartphones, "
    "UCI Machine Learning Repository, DOI 10.24432/C54S4K. "
    "The included demo data is synthetic and exists only for instant offline execution."
)
