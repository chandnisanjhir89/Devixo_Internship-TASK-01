import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Students Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------
# Load & Clean Data
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

df = load_data()

# ----------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------
st.sidebar.header("🔍 Filters")

gender_filter = st.sidebar.multiselect(
    "Gender", options=df["gender"].unique(), default=list(df["gender"].unique())
)
lunch_filter = st.sidebar.multiselect(
    "Lunch Type", options=df["lunch"].unique(), default=list(df["lunch"].unique())
)
prep_filter = st.sidebar.multiselect(
    "Test Preparation Course",
    options=df["test_preparation_course"].unique(),
    default=list(df["test_preparation_course"].unique())
)
race_filter = st.sidebar.multiselect(
    "Race/Ethnicity Group",
    options=sorted(df["race/ethnicity"].unique()),
    default=sorted(df["race/ethnicity"].unique())
)

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["lunch"].isin(lunch_filter)) &
    (df["test_preparation_course"].isin(prep_filter)) &
    (df["race/ethnicity"].isin(race_filter))
]

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.title("📊 Students Performance Dashboard")
st.markdown("Interactive dashboard for **Task 01 – Devixo Solutions AI/ML Internship Program**")
st.markdown("---")

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Records", filtered_df.shape[0])
col2.metric("Total Columns", filtered_df.shape[1])
col3.metric("Avg Math Score", round(filtered_df["math_score"].mean(), 2) if len(filtered_df) else "-")
col4.metric("Avg Reading Score", round(filtered_df["reading_score"].mean(), 2) if len(filtered_df) else "-")
col5.metric("Avg Writing Score", round(filtered_df["writing_score"].mean(), 2) if len(filtered_df) else "-")

st.markdown("---")

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust the filters.")
    st.stop()

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
with st.expander("📄 View Filtered Dataset"):
    st.dataframe(filtered_df)

# ----------------------------------------------------
# Summary Statistics
# ----------------------------------------------------
st.subheader("📈 Summary Statistics")
st.dataframe(filtered_df[["math_score", "reading_score", "writing_score"]].describe())

st.markdown("---")

# ----------------------------------------------------
# Charts Row 1
# ----------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Average Score by Subject")
    avg_scores = filtered_df[["math_score", "reading_score", "writing_score"]].mean()
    fig, ax = plt.subplots()
    avg_scores.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Average Score")
    ax.set_xlabel("Subject")
    st.pyplot(fig)

with c2:
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    filtered_df["gender"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax, ylabel=""
    )
    st.pyplot(fig)

# ----------------------------------------------------
# Charts Row 2
# ----------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Math vs Reading Score")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["math_score"], filtered_df["reading_score"], alpha=0.6)
    ax.set_xlabel("Math Score")
    ax.set_ylabel("Reading Score")
    st.pyplot(fig)

with c4:
    st.subheader("Score Distribution (Box Plot)")
    fig, ax = plt.subplots()
    filtered_df[["math_score", "reading_score", "writing_score"]].plot(kind="box", ax=ax)
    st.pyplot(fig)

# ----------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(
    filtered_df[["math_score", "reading_score", "writing_score"]].corr(),
    annot=True, cmap="Blues", ax=ax
)
st.pyplot(fig)

st.markdown("---")

# ----------------------------------------------------
# Test Preparation Impact
# ----------------------------------------------------
st.subheader("🎯 Test Preparation Course Impact")
prep_avg = filtered_df.groupby("test_preparation_course")[
    ["math_score", "reading_score", "writing_score"]
].mean()
st.bar_chart(prep_avg)

# ----------------------------------------------------
# Key Insights
# ----------------------------------------------------
st.markdown("---")
st.subheader("💡 Key Insights")

corr_rw = filtered_df["reading_score"].corr(filtered_df["writing_score"])
best_subject = filtered_df[["math_score", "reading_score", "writing_score"]].mean().idxmax()
most_common_group = filtered_df["race/ethnicity"].value_counts().idxmax()

st.markdown(f"""
- **Reading & Writing correlation:** `{corr_rw:.2f}` — the strongest relationship among the three subjects.
- **Highest performing subject (on average):** `{best_subject.replace('_', ' ').title()}`
- **Most common race/ethnicity group:** `{most_common_group}`
- Students who completed the **test preparation course** generally score higher across all subjects (see chart above).
- Use the sidebar filters to explore how gender, lunch type, and preparation course affect performance.
""")

st.markdown("---")
st.caption("Built with Streamlit · Devixo Solutions AI/ML Internship Program · Task 01 Bonus")