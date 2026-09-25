from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="AVIP 2026 | Titanic EDA",
    page_icon=":material/directions_boat:",
    layout="wide",
)

RAW_DATA_PATH = Path("data/titanic.csv")
CLEANED_DATA_PATH = Path("data/titanic_cleaned.csv")
NUMERIC_FEATURES = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Titanic exploratory data analysis")
st.write(
    "An interactive overview of passenger survival patterns and demographic features, "
    "built from the Titanic data included with the AVIP 2026 Task 2 project."
)

missing_files = [path for path in (RAW_DATA_PATH, CLEANED_DATA_PATH) if not path.is_file()]
if missing_files:
    st.error(
        "Required project data file(s) were not found: "
        + ", ".join(path.as_posix() for path in missing_files)
        + ". Run this app from the repository root."
    )
    st.stop()

raw_df = load_csv(RAW_DATA_PATH.as_posix())
df_clean = load_csv(CLEANED_DATA_PATH.as_posix())

required_columns = {"survived", "pclass", "sex", "age", *NUMERIC_FEATURES}
missing_columns = required_columns.difference(df_clean.columns)
if missing_columns:
    st.error(
        "The cleaned dataset is missing required analysis columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

st.header("Dataset overview")
overview_source, overview_analysis = st.columns(2)
with overview_source:
    st.metric("Passengers in source dataset", f"{raw_df.shape[0]:,}", border=True)
    st.metric("Source dataset columns", f"{raw_df.shape[1]:,}", border=True)
with overview_analysis:
    st.metric("Passengers in cleaned analysis", f"{df_clean.shape[0]:,}", border=True)
    st.metric("Columns in cleaned analysis", f"{df_clean.shape[1]:,}", border=True)

missing_summary = (
    raw_df.isna()
    .sum()
    .rename("missing_values")
    .to_frame()
    .assign(percent_missing=lambda summary: summary["missing_values"].div(len(raw_df)).mul(100))
)
st.subheader("Missing values in the source dataset")
st.dataframe(
    missing_summary.style.format({"percent_missing": "{:.1f}%"}),
    width="stretch",
)

survivor_count = int(df_clean["survived"].sum())
passenger_count = len(df_clean)
overall_survival_rate = df_clean["survived"].mean()

st.header("Survival overview")
with st.container(horizontal=True):
    st.metric("Total passengers", f"{passenger_count:,}", border=True)
    st.metric("Survivors", f"{survivor_count:,}", border=True)
    st.metric("Overall survival rate", f"{overall_survival_rate:.1%}", border=True)

survival_by_class = (
    df_clean.groupby(["pclass", "class"], as_index=False, observed=True)
    .agg(passengers=("survived", "size"), survivors=("survived", "sum"), survival_rate=("survived", "mean"))
    .sort_values("pclass")
)
survival_by_class["survival_rate_percent"] = survival_by_class["survival_rate"] * 100

survival_by_gender = (
    df_clean.groupby("sex", as_index=False, observed=True)
    .agg(passengers=("survived", "size"), survivors=("survived", "sum"), survival_rate=("survived", "mean"))
    .sort_values("sex")
)
survival_by_gender["survival_rate_percent"] = survival_by_gender["survival_rate"] * 100

st.header("Survival by passenger group")
class_column, gender_column = st.columns(2)
with class_column:
    st.subheader("Survival rate by passenger class")
    class_fig, class_ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=survival_by_class,
        x="class",
        y="survival_rate_percent",
        color="#4C78A8",
        ax=class_ax,
    )
    class_ax.set(title="Survival rate by passenger class", xlabel="Passenger class", ylabel="Survival rate (%)")
    class_ax.set_ylim(0, 100)
    for container in class_ax.containers:
        class_ax.bar_label(container, fmt="%.1f%%", padding=3)
    class_fig.tight_layout()
    st.pyplot(class_fig)
    plt.close(class_fig)
    st.dataframe(
        survival_by_class[["class", "passengers", "survivors", "survival_rate_percent"]].rename(
            columns={"class": "passenger_class", "survival_rate_percent": "survival_rate (%)"}
        ).style.format({"survival_rate (%)": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )

with gender_column:
    st.subheader("Survival rate by gender")
    gender_fig, gender_ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=survival_by_gender,
        x="sex",
        y="survival_rate_percent",
        color="#F58518",
        ax=gender_ax,
    )
    gender_ax.set(title="Survival rate by gender", xlabel="Gender", ylabel="Survival rate (%)")
    gender_ax.set_ylim(0, 100)
    for container in gender_ax.containers:
        gender_ax.bar_label(container, fmt="%.1f%%", padding=3)
    gender_fig.tight_layout()
    st.pyplot(gender_fig)
    plt.close(gender_fig)
    st.dataframe(
        survival_by_gender[["sex", "passengers", "survivors", "survival_rate_percent"]].rename(
            columns={"survival_rate_percent": "survival_rate (%)"}
        ).style.format({"survival_rate (%)": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )

st.header("Age distribution")
age_fig, age_ax = plt.subplots(figsize=(10, 4.5))
sns.histplot(data=df_clean, x="age", bins=20, color="#54A24B", edgecolor="white", ax=age_ax)
age_ax.set(title="Passenger age distribution", xlabel="Age (years)", ylabel="Passenger count")
age_fig.tight_layout()
st.pyplot(age_fig)
plt.close(age_fig)
st.caption(
    f"Based on {df_clean['age'].count():,} cleaned age values; "
    f"median age: {df_clean['age'].median():.2f} years."
)

st.header("Correlation heatmap")
available_numeric_features = [column for column in NUMERIC_FEATURES if column in df_clean.columns]
correlation_matrix = df_clean[available_numeric_features].corr()
heatmap_fig, heatmap_ax = plt.subplots(figsize=(9, 6.5))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=heatmap_ax,
)
heatmap_ax.set_title("Correlation between numeric features")
heatmap_fig.tight_layout()
st.pyplot(heatmap_fig)
plt.close(heatmap_fig)

highest_class = survival_by_class.loc[survival_by_class["survival_rate"].idxmax()]
lowest_class = survival_by_class.loc[survival_by_class["survival_rate"].idxmin()]
class_rate_gap = (highest_class["survival_rate"] - lowest_class["survival_rate"]) * 100
highest_gender = survival_by_gender.loc[survival_by_gender["survival_rate"].idxmax()]
lowest_gender = survival_by_gender.loc[survival_by_gender["survival_rate"].idxmin()]
gender_rate_gap = (highest_gender["survival_rate"] - lowest_gender["survival_rate"]) * 100

st.header("Key observations")
st.markdown(
    f"- The highest class survival rate was **{highest_class['class']}** "
    f"({highest_class['survival_rate']:.1%}), compared with "
    f"{lowest_class['class']} ({lowest_class['survival_rate']:.1%}); "
    f"the observed gap was {class_rate_gap:.1f} percentage points."
)
st.markdown(
    f"- The highest gender survival rate was **{highest_gender['sex']}** "
    f"({highest_gender['survival_rate']:.1%}), compared with "
    f"{lowest_gender['sex']} ({lowest_gender['survival_rate']:.1%}); "
    f"the observed gap was {gender_rate_gap:.1f} percentage points."
)
st.markdown(
    f"- The cleaned age distribution has a median of **{df_clean['age'].median():.2f} years** "
    f"across {df_clean['age'].count():,} passenger records."
)

survival_correlations = correlation_matrix["survived"].drop(labels="survived")
strongest_survival_feature = survival_correlations.abs().idxmax()
strongest_survival_correlation = survival_correlations[strongest_survival_feature]
st.markdown(
    f"- Among the displayed numeric features, **{strongest_survival_feature}** had the largest "
    f"absolute correlation with survival (r = {strongest_survival_correlation:.2f}); "
    "this is an association, not evidence of causation."
)

st.header("Conclusion")
st.write(
    f"In the cleaned analysis, {survivor_count:,} of {passenger_count:,} passengers survived "
    f"({overall_survival_rate:.1%}). Survival rates varied across the passenger-class and gender "
    "groups represented in the data. The age distribution and numeric correlations provide "
    "additional descriptive context. These results summarize this dataset and do not establish "
    "causal relationships."
)

st.header("Download cleaned dataset")
st.download_button(
    label="Download titanic_cleaned.csv",
    data=CLEANED_DATA_PATH.read_bytes(),
    file_name="titanic_cleaned.csv",
    mime="text/csv",
)
