"""
Healthcare ML Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")

# ── Helper: load CSVs ────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# Paths – adjust if your files are in a subfolder
MODEL1_PATH   = "model1_comparison.csv"
MODEL2_PATH   = "model2_comparison.csv"
CL_PATH       = "cl_model_comparison.csv"
HIST_PATH     = "historical_data.csv"
CURR_PATH     = "current_data.csv"

# Plot image paths (put all .jpeg files next to app.py)
PLOTS = {
    "ROC – MLP (Dataset 2)"         : "mlproc2.jpeg",
    "ROC – SVM (Dataset 2)"         : "svmroc2.jpeg",
    "ROC – Decision Tree (Dataset 2)": "dtroc2.jpeg",
    "ROC – MLP (Continual Learning)" : "mlproc_cl.jpeg",
    "ROC – SVM (Continual Learning)" : "svmroc_cl.jpeg",
    "ROC – DT (Continual Learning)"  : "dtroc_cl.jpeg",
    "Feature Importance – DT"        : "impfeatdt2.jpeg",
    "Feature Importance – MLP"       : "featimpmlp2.jpeg",
    "Feature Importance – SVM"       : "featimpsvm2.jpeg",
    "Num Conditions vs Age"          : "numcond.jpeg",
    "Age Distribution Comparison"    : "agecomp.jpeg",
    "Marital Status Comparison"      : "maritalcomp.jpeg",
}

# Numeric features used throughout the app
NUMERIC_FEATURES = [
    "AGE", "num_encounters", "mean_encounter_cost", "total_claim_cost",
    "avg_payer_coverage", "num_conditions", "medication_cost",
    "enc_ambulatory_count", "enc_emergency_count", "enc_wellness_count",
    "days_since_last_enc", "obs_bmi_mean", "obs_systolic_bp_mean",
    "obs_diastolic_bp_mean", "obs_weight_mean", "obs_glucose_mean",
    "HEALTHCARE_EXPENSES", "INCOME", "medication_count",
]

# ── Load data ────────────────────────────────────────────────────────────────
m1  = load_csv(MODEL1_PATH)
m2  = load_csv(MODEL2_PATH)
cl  = load_csv(CL_PATH)
df_hist = load_csv(HIST_PATH)
df_curr = load_csv(CURR_PATH)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "EDA", "Feature Importance", "Drift Analysis"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Healthcare ML Dashboard")
    st.markdown(
        """
        This dashboard summarises a machine-learning study on a **synthetic healthcare dataset**
        (Massachusetts, Synthea).  Three classifiers — **Decision Tree, SVM, and Neural Network (MLP)** —
        were trained on a historical cohort and later updated with continual learning on a newer cohort.

        Use the sidebar to navigate between sections.
        """
    )

    st.subheader("Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Records",  f"{len(df_hist):,}")
    col2.metric("Current Records",     f"{len(df_curr):,}")
    col3.metric("Features",            len(NUMERIC_FEATURES))
    col4.metric("Target",              "DECEASED")

    st.subheader("Best Model per Dataset")
    summary = pd.DataFrame({
        "Dataset"        : ["Historical (D1)", "Current (D2)", "After Continual Learning"],
        "Best Model"     : ["Decision Tree",   "Decision Tree", "Decision Tree"],
        "F1 Score"       : [0.714,             0.764,           0.729],
        "AUC"            : ["—",               0.83,            0.86],
    })
    st.table(summary)

    st.info(
        "**Key finding:** Decision Tree consistently outperforms SVM and MLP on this dataset. "
        "Continual learning lifted the SVM AUC from 0.51 → 0.69 and MLP from 0.60 → 0.69."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")

    tab1, tab2, tab3 = st.tabs(
        ["Dataset 1 (Historical)", "Dataset 2 (Current)", "Continual Learning"]
    )

    def show_metrics(df, title, metric="F1"):
        st.subheader(f"Metrics Table – {title}")
        fmt_cols = [c for c in df.columns if c != "Model"]
        styled = df.style.format({c: "{:.3f}" for c in fmt_cols}).highlight_max(
            subset=fmt_cols, color="#d4edda", axis=0
        )
        st.dataframe(styled, use_container_width=True)

        st.subheader(f"{metric} Score Comparison – {title}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df["Model"], df[metric], color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric} by Model – {title}")
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab1:
        show_metrics(m1, "Historical Dataset", "F1")

    with tab2:
        show_metrics(m2, "Current Dataset", "F1")
        st.subheader("ROC Curves – Dataset 2")
        c1, c2, c3 = st.columns(3)
        for col, key in zip(
            [c1, c2, c3],
            ["ROC – Decision Tree (Dataset 2)", "ROC – SVM (Dataset 2)", "ROC – MLP (Dataset 2)"],
        ):
            if os.path.exists(PLOTS[key]):
                col.image(PLOTS[key], use_container_width=True, caption=key)

    with tab3:
        show_metrics(cl, "After Continual Learning", "F1")
        st.subheader("ROC Curves – Before vs After Continual Learning")
        c1, c2, c3 = st.columns(3)
        for col, key in zip(
            [c1, c2, c3],
            ["ROC – DT (Continual Learning)", "ROC – SVM (Continual Learning)", "ROC – MLP (Continual Learning)"],
        ):
            if os.path.exists(PLOTS[key]):
                col.image(PLOTS[key], use_container_width=True, caption=key)

        # AUC comparison before / after CL (hard-coded from the plots)
        st.subheader("AUC Before vs After Continual Learning")
        auc_data = pd.DataFrame({
            "Model"  : ["Decision Tree", "SVM",  "MLP"],
            "Before" : [0.84,            0.51,   0.60],
            "After"  : [0.86,            0.69,   0.69],
        })
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.arange(len(auc_data))
        ax.bar(x - 0.2, auc_data["Before"], 0.35, label="Before CL", color="#4C72B0")
        ax.bar(x + 0.2, auc_data["After"],  0.35, label="After CL",  color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(auc_data["Model"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC")
        ax.set_title("AUC Before vs After Continual Learning")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    # --- Target distribution (class imbalance) --------------------------------
    st.subheader("Class Distribution – DECEASED (Target)")
    col1, col2 = st.columns(2)
    for df_plot, label, col in [
        (df_hist, "Historical", col1), (df_curr, "Current", col2)
    ]:
        counts = df_plot["DECEASED"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Alive (0)", "Deceased (1)"], counts.values, color=["#4C72B0", "#DD8452"])
        ax.set_title(f"Target – {label}")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(v), ha="center")
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()

    pct_hist = df_hist["DECEASED"].mean() * 100
    pct_curr = df_curr["DECEASED"].mean() * 100
    st.info(
        f"**Class imbalance:** Historical → {pct_hist:.1f}% deceased | "
        f"Current → {pct_curr:.1f}% deceased"
    )

    # --- Age distribution -------------------------------------------------------
    st.subheader("Age Distribution – Historical vs Current")
    if os.path.exists(PLOTS["Age Distribution Comparison"]):
        st.image(PLOTS["Age Distribution Comparison"], use_container_width=True)

    # --- Num conditions vs age --------------------------------------------------
    st.subheader("Number of Conditions vs Age")
    if os.path.exists(PLOTS["Num Conditions vs Age"]):
        st.image(PLOTS["Num Conditions vs Age"], use_container_width=True)

    # --- Marital status ---------------------------------------------------------
    st.subheader("Marital Status Distribution")
    if os.path.exists(PLOTS["Marital Status Comparison"]):
        st.image(PLOTS["Marital Status Comparison"], use_container_width=True)

    # --- Feature histogram (interactive) ----------------------------------------
    st.subheader("Feature Histogram Explorer")
    feat = st.selectbox("Select feature", NUMERIC_FEATURES)
    fig, ax = plt.subplots(figsize=(7, 3))
    for df_plot, label, color in [
        (df_hist, "Historical", "#4C72B0"),
        (df_curr, "Current",    "#DD8452"),
    ]:
        col = df_plot[feat].dropna()
        col = col[col.between(col.quantile(0.01), col.quantile(0.99))]  # clip outliers
        ax.hist(col, bins=30, alpha=0.6, label=label, color=color)
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feat}")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Correlation heatmap (top numeric features) ----------------------------
    st.subheader("Correlation Heatmap – Top Numeric Features (Historical)")
    TOP_FEATS = [
        "AGE", "num_encounters", "mean_encounter_cost", "num_conditions",
        "medication_cost", "obs_systolic_bp_mean", "obs_diastolic_bp_mean",
        "obs_bmi_mean", "obs_weight_mean", "DECEASED",
    ]
    corr = df_hist[TOP_FEATS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TOP_FEATS)))
    ax.set_yticks(range(len(TOP_FEATS)))
    ax.set_xticklabels(TOP_FEATS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(TOP_FEATS, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    for i in range(len(TOP_FEATS)):
        for j in range(len(TOP_FEATS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.title("Feature Importance")
    st.markdown("Top features identified by each model on Dataset 2.")

    c1, c2, c3 = st.columns(3)
    for col, key in zip(
        [c1, c2, c3],
        ["Feature Importance – DT", "Feature Importance – MLP", "Feature Importance – SVM"],
    ):
        if os.path.exists(PLOTS[key]):
            col.image(PLOTS[key], use_container_width=True, caption=key)

    st.subheader("Decision Tree – Top 10 Features (Summary Table)")
    dt_feats = pd.DataFrame({
        "Feature"   : [
            "obs_systolic_bp_latest", "obs_diastolic_bp_latest", "AGE",
            "mean_encounter_cost", "obs_systolic_bp_mean", "enc_ambulatory_count",
            "proc_cost_var", "num_encounters", "INCOME", "obs_diastolic_bp_var",
        ],
        "Importance": [0.52, 0.17, 0.06, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015],
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(dt_feats["Feature"][::-1], dt_feats["Importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Decision Tree – Top 10 Features")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Drift Analysis":
    st.title("Drift Analysis – Historical vs Current Dataset")
    st.markdown(
        "Comparing feature distributions to detect **data drift** between the historical "
        "training cohort (D1) and the current cohort (D2)."
    )

    # --- Mean comparison table ------------------------------------------------
    st.subheader("Feature Mean Comparison")
    avail = [f for f in NUMERIC_FEATURES if f in df_hist.columns and f in df_curr.columns]
    means_hist = df_hist[avail].mean()
    means_curr = df_curr[avail].mean()
    drift_df = pd.DataFrame({
        "Feature"         : avail,
        "Mean (Historical)": means_hist.values.round(3),
        "Mean (Current)"  : means_curr.values.round(3),
    })
    drift_df["Absolute Diff"] = (drift_df["Mean (Current)"] - drift_df["Mean (Historical)"]).abs().round(3)
    drift_df["% Change"]      = (
        (drift_df["Mean (Current)"] - drift_df["Mean (Historical)"]) /
        (drift_df["Mean (Historical)"].replace(0, np.nan)) * 100
    ).round(1)
    drift_df = drift_df.sort_values("Absolute Diff", ascending=False).reset_index(drop=True)
    st.dataframe(drift_df, use_container_width=True)

    # --- Top 10 drifted features chart ----------------------------------------
    st.subheader("Top 10 Features with Largest Mean Shift")
    top10 = drift_df.head(10)
    x = np.arange(len(top10))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, top10["Mean (Historical)"], 0.35, label="Historical", color="#4C72B0")
    ax.bar(x + 0.2, top10["Mean (Current)"],    0.35, label="Current",    color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(top10["Feature"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Value")
    ax.set_title("Top 10 Drifted Features")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Uploaded drift plots --------------------------------------------------
    st.subheader("Distribution Plots")
    c1, c2 = st.columns(2)
    if os.path.exists(PLOTS["Age Distribution Comparison"]):
        c1.image(PLOTS["Age Distribution Comparison"], use_container_width=True, caption="Age – Historical vs Current")
    if os.path.exists(PLOTS["Marital Status Comparison"]):
        c2.image(PLOTS["Marital Status Comparison"], use_container_width=True, caption="Marital – Historical vs Current")
    if os.path.exists(PLOTS["Num Conditions vs Age"]):
        st.image(PLOTS["Num Conditions vs Age"], use_container_width=True, caption="Num Conditions vs Age")