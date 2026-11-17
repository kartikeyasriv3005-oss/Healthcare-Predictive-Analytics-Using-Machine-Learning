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
    # ✦ NEW: Added "Summary & Findings" page to navigation
    ["Overview", "Model Performance", "EDA", "Feature Importance", "Drift Analysis", "Summary & Findings"],
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
 
 
# ✦ NEW ══════════════════════════════════════════════════════════════════════════
# 6. SUMMARY & FINDINGS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Summary & Findings":
    st.title("Summary of Findings")
    st.markdown(
        "A comprehensive summary of the study covering feature importance, model complexity, "
        "bias-variance trade-offs, temporal filtering effects, and the continual learning strategy."
    )
 
    # ── A. Feature Importance ────────────────────────────────────────────────────
    st.header("1. Feature Importance Analysis")
    col1, col2, col3 = st.columns(3)
    for col, key in zip(
        [col1, col2, col3],
        ["Feature Importance – DT", "Feature Importance – MLP", "Feature Importance – SVM"],
    ):
        if os.path.exists(PLOTS[key]):
            col.image(PLOTS[key], use_container_width=True, caption=key)
 
    st.markdown("""
    | Model | Top Features | Key Insight |
    |-------|-------------|-------------|
    | **Decision Tree** | `obs_systolic_bp_latest` (0.52), `obs_diastolic_bp_latest` (0.17), `AGE` (0.06) | BP readings dominate; latest readings outweigh averages |
    | **MLP** | `obs_glucose_var`, `med_unique`, `obs_diastolic_bp_latest`, `med_active_count` | Variability & medication diversity are strong signals |
    | **SVM** | `obese_flag` (0.0067), `days_since_last_enc` (0.006), BP metrics | Recency of care and obesity flag are leading indicators |
 
    **Insight:** Blood pressure (especially *latest* readings) and medication complexity are the most
    consistent predictors of patient mortality across all three models. The DT is highly concentrated
    on a single feature (`obs_systolic_bp_latest`), while MLP and SVM spread importance more broadly.
    """)
 
    # ── B. Model Complexity & Bias-Variance ──────────────────────────────────────
    st.header("2. Model Complexity & Bias-Variance Trade-offs")
 
    tab_dt, tab_svm, tab_mlp = st.tabs(["Decision Tree", "SVM", "MLP"])
 
    with tab_dt:
        st.subheader("Decision Tree – Depth Analysis")
        dt_depth = pd.DataFrame({
            "Max Depth"      : [2, 3, 4, 5, "Full (default)"],
            "Train Accuracy" : [0.9852, 0.9926, 0.9981, 1.0000, 1.0000],
            "Test Accuracy"  : [0.9778, 0.9630, 0.9704, 0.9704, 0.9704],
        })
        st.dataframe(dt_depth, use_container_width=True)
 
        fig, ax = plt.subplots(figsize=(6, 3))
        depths_x = [2, 3, 4, 5, 6]
        train_acc = [0.9852, 0.9926, 0.9981, 1.000, 1.000]
        test_acc  = [0.9778, 0.9630, 0.9704, 0.9704, 0.9704]
        ax.plot(depths_x, train_acc, "o-", label="Train", color="#4C72B0")
        ax.plot(depths_x, test_acc,  "s--", label="Test",  color="#DD8452")
        ax.set_xlabel("Max Depth"); ax.set_ylabel("Accuracy")
        ax.set_title("DT Accuracy vs Tree Depth"); ax.legend(); ax.set_ylim(0.95, 1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
        st.info(
            "**Bias-Variance:** At depth 2 the model slightly underfits (highest test acc but low train). "
            "At depth ≥ 5 training accuracy hits 1.0 while test accuracy plateaus at 0.970, indicating "
            "mild overfitting. Depth 4–5 is the sweet spot."
        )
 
    with tab_svm:
        st.subheader("SVM – Regularisation (C) and Kernel (γ) Analysis")
        svm_c = pd.DataFrame({
            "C"              : [0.1, 1, 10, 100, 1000],
            "Train Accuracy" : [0.956, 0.978, 1.000, 1.000, 1.000],
            "Test Accuracy"  : [0.948, 0.948, 0.948, 0.956, 0.956],
        })
        svm_g = pd.DataFrame({
            "γ (gamma)"      : [0.01, 0.1, 1, 10],
            "Train Accuracy" : [0.967, 0.994, 1.000, 1.000],
            "Test Accuracy"  : [0.948, 0.948, 0.948, 0.948],
        })
        col_a, col_b = st.columns(2)
        col_a.markdown("**C sweep (γ fixed)**"); col_a.dataframe(svm_c, use_container_width=True)
        col_b.markdown("**γ sweep (C fixed)**"); col_b.dataframe(svm_g, use_container_width=True)
 
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for ax, df_s, x_col, title, log in zip(
            axes, [svm_c, svm_g], ["C", "γ (gamma)"],
            ["Train vs Test Accuracy by C", "Train vs Test Accuracy by γ"], [True, False]
        ):
            xs = range(len(df_s))
            ax.plot(xs, df_s["Train Accuracy"], "o-", label="Train", color="#4C72B0")
            ax.plot(xs, df_s["Test Accuracy"],  "s--", label="Test",  color="#DD8452")
            ax.set_xticks(xs); ax.set_xticklabels(df_s[x_col], fontsize=8)
            ax.set_ylim(0.93, 1.01); ax.set_title(title); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
        st.info(
            "**Bias-Variance:** Test accuracy is largely insensitive to both C and γ, "
            "suggesting the SVM has reached its capacity on this data. High C / γ values "
            "cause training accuracy to reach 1.0 (overfitting) while test accuracy stagnates."
        )
 
    with tab_mlp:
        st.subheader("MLP – Hidden Layer Architecture Analysis")
        mlp_layers = pd.DataFrame({
            "Architecture"   : ["(2,)", "(4,)", "(8,)", "(8,4)", "(12,6)"],
            "Train Accuracy" : [0.9574, 0.9926, 0.9963, 0.9889, 0.9963],
            "Test Accuracy"  : [0.9481, 0.9407, 0.9407, 0.9333, 0.9259],
        })
        st.dataframe(mlp_layers, use_container_width=True)
 
        fig, ax = plt.subplots(figsize=(6, 3))
        xs = range(len(mlp_layers))
        ax.plot(xs, mlp_layers["Train Accuracy"], "o-", label="Train", color="#4C72B0")
        ax.plot(xs, mlp_layers["Test Accuracy"],  "s--", label="Test",  color="#DD8452")
        ax.set_xticks(xs); ax.set_xticklabels(mlp_layers["Architecture"], fontsize=9)
        ax.set_ylim(0.90, 1.01); ax.set_ylabel("Accuracy")
        ax.set_title("MLP Accuracy by Architecture"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
        st.info(
            "**Bias-Variance:** As network depth/width grows, training accuracy improves but "
            "test accuracy *decreases*, a classic overfitting pattern. The simplest architecture `(2,)` "
            "achieves the best test accuracy. More complex MLPs may require regularisation (dropout, L2)."
        )
 
    # ── C. Confusion Matrices ────────────────────────────────────────────────────
    st.header("3. Confusion Matrix Analysis")
    st.markdown(
        "Confusion matrices reveal *how* each model errs — particularly important given the "
        "class imbalance (few deceased patients) in this dataset."
    )
 
    # ── Confusion matrix image paths (uploaded by user) ──────────────────────
    CM_IMAGES = {
        "DT_D1"  : "agecomp.jpeg",  # Placeholder: replace with actual confusion matrix image for DT on D1
        "DT_D2"  : "cm_d2_dt.jpeg",
        "SVM_D1" : "cm_d1_svm.jpeg",
        "SVM_D2" : "cm_d2_svm.jpeg",
        "MLP_D1" : "cm_d1_mlp.jpeg",
        "MLP_D2" : "cm_d2_mlp.jpeg",
    }
 
    # st.subheader("Decision Tree")
    # c1, c2 = st.columns(2)
    # for col, key in zip([c1, c2], ["DT_D1", "DT_D2"]):
    #     if os.path.exists(CM_IMAGES[key]):
    #         col.image(CM_IMAGES[key], use_container_width=True, caption=key)
 
    # st.subheader("SVM")
    # c1, c2 = st.columns(2)
    # for col, key in zip([c1, c2], ["SVM_D1", "SVM_D2"]):
    #     if os.path.exists(CM_IMAGES[key]):
    #         col.image(CM_IMAGES[key], use_container_width=True, caption=key)
 
    # st.subheader("MLP")
    # c1, c2 = st.columns(2)
    # for col, key in zip([c1, c2], ["MLP_D1", "MLP_D2"]):
    #     if os.path.exists(CM_IMAGES[key]):
    #         col.image(CM_IMAGES[key], use_container_width=True, caption=key)
 
    # Confusion matrix summary table
    cm_summary = pd.DataFrame({
        "Model-Dataset": ["DT_D1", "DT_D2", "SVM_D1", "SVM_D2", "MLP_D1", "MLP_D2"],
        "TN"  : [126, 223, 127, 283, 124, 215],
        "FP"  : [2,   61,  1,   1,   4,   69],
        "FN"  : [2,   18,  6,   144, 5,   96],
        "TP"  : [5,   128, 1,   2,   2,   50],
        "Notes": [
            "High precision, misses some deceased",
            "Good balance; moderate FP & FN",
            "Near-perfect specificity; very low recall",
            "Predicts almost all as negative; extreme FN",
            "Low recall on deceased class",
            "Better recall but high FP/FN counts",
        ],
    })
    st.dataframe(cm_summary, use_container_width=True)
 
    st.warning(
        "**Key observation:** The SVM on D2 produces 144 False Negatives — it predicts the "
        "vast majority of deceased patients as alive. This is a critical failure in a clinical context. "
        "The Decision Tree on D2 achieves a much better balance (18 FN, 61 FP), making it the safer choice."
    )
 
    # ── D. Temporal Filtering & Feature Representation ───────────────────────────
    st.header("4. Temporal Filtering & Feature Representation")
    st.markdown("""
    - **Latest vs. Mean features:** The Decision Tree heavily favours `obs_systolic_bp_latest` 
      and `obs_diastolic_bp_latest` over their mean counterparts, demonstrating that *recency* 
      of clinical observations is a stronger mortality signal than historical averages.
    - **Temporal drift (age distribution):** The current cohort skews younger (~25–30 peak) 
      versus the historical cohort (~50–55 peak), reflecting a different patient population 
      captured in the newer data window. This shift explains why models trained on D1 need 
      updating to remain accurate.
    - **Feature variability:** MLP relies strongly on `obs_glucose_var` — the *variance* of 
      glucose readings — suggesting that fluctuation in glucose levels is more predictive of 
      outcomes than its average value alone.
    - **Days since last encounter (`days_since_last_enc`):** The SVM's second most important 
      feature, indicating that patients who haven't been seen recently have different risk profiles.
    """)
 
    # ── E. Continual Learning Strategy ───────────────────────────────────────────
    st.header("5. Continual Learning Strategy – Conclusion")
 
    auc_df = pd.DataFrame({
        "Model"  : ["Decision Tree", "SVM",  "MLP"],
        "Before CL (AUC)" : [0.84, 0.51, 0.60],
        "After CL (AUC)"  : [0.86, 0.69, 0.69],
        "AUC Gain"        : ["+0.02", "+0.18", "+0.09"],
    })
    st.dataframe(auc_df, use_container_width=True)
 
    col_roc1, col_roc2 = st.columns(2)
    if os.path.exists(PLOTS["ROC – DT (Continual Learning)"]):
        col_roc1.image(PLOTS["ROC – DT (Continual Learning)"], use_container_width=True,
                       caption="DT ROC – Before vs After CL")
 
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(3)
    ax.bar(x - 0.2, [0.84, 0.51, 0.60], 0.35, label="Before CL", color="#4C72B0")
    ax.bar(x + 0.2, [0.86, 0.69, 0.69], 0.35, label="After CL",  color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(["Decision Tree", "SVM", "MLP"])
    ax.set_ylim(0, 1); ax.set_ylabel("AUC"); ax.set_title("AUC Before vs After CL"); ax.legend()
    plt.tight_layout()
    col_roc2.pyplot(fig); plt.close()
 
    st.success("""
    **Conclusion – Continual Learning was Successful:**
 
    - ✅ **Decision Tree** remained the best overall model (AUC 0.84 → 0.86, F1 maintained).
    - ✅ **SVM** showed the largest AUC gain (+0.18), recovering from near-random performance on D2.
    - ✅ **MLP** improved meaningfully (+0.09 AUC), though its confusion matrix still shows room 
      for improvement in recall on the deceased class.
    - ⚠️ Despite AUC gains, SVM continues to struggle with recall on D2 (144 False Negatives), 
      suggesting it needs stronger class-weighting or feature engineering to handle the minority class.
    - 📌 The continual learning strategy — incrementally updating models on the current cohort — 
      successfully mitigated the performance degradation caused by temporal data drift, validating 
      its use in evolving clinical datasets.
    """)
 
    st.header("6. Overall Recommendation")
    st.markdown("""
    | Criterion | Recommended Model |
    |-----------|------------------|
    | Best overall accuracy | **Decision Tree** |
    | Best AUC after CL | **Decision Tree** (0.86) |
    | Best recall on deceased class (D2) | **Decision Tree** (lowest FN = 18) |
    | Most interpretable | **Decision Tree** |
    | Most improved by CL | **SVM** (+0.18 AUC gain) |
 
    > **Final recommendation:** The **Decision Tree** is the most suitable model for this clinical 
    > mortality prediction task. It is interpretable, maintains a strong balance between precision 
    > and recall, and responds well to continual learning. Future work should focus on addressing 
    > class imbalance (SMOTE / class weights) and exploring ensemble methods (Random Forest, Gradient Boosting).
    """)