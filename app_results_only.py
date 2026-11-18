"""H-MOOA IDS - results viewer (no trained models needed).
Run:  streamlit run app_results_only.py
"""
import os
import pandas as pd, streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
st.set_page_config(page_title="H-MOOA IDS Results", page_icon="🛡️", layout="wide")
st.title("🛡️ H-MOOA Intrusion Detection - Research Results")
st.caption("CIC-IIoT-2025 | 685,671 flows | 8 classes | 78 features reduced to 15 by H-MOOA")

def show(fn, title, note=""):
    st.subheader(title)
    if note: st.caption(note)
    d = pd.read_csv(os.path.join(R, fn)); st.dataframe(d, use_container_width=True); return d

d = show("final_results.csv", "Baselines vs H-MOOA", "DT / RF / SVM / LSTM on all features, BGWO and H-MOOA features, plus the fusion model")
k = st.selectbox("Plot metric", [c for c in d.columns if c != "Model"], index=list(d.columns).index("F1") - 1)
st.bar_chart(d.set_index("Model")[k])
o = show("optimizer_comparison.csv", "Optimizer comparison (H-MOOA, NSGA-II, PSO, BGWO, Bat, SSO)")
st.bar_chart(o.set_index("Optimizer")["HV_mean"])
show("optimizer_wilcoxon.csv", "Wilcoxon test, H-MOOA vs competitors (Holm corrected)")
show("cv_statistics.csv", "Repeated cross-validation statistics")
z = show("zero_day.csv", "Unknown-attack detection (leave-one-class-out)")
st.bar_chart(z.set_index("unseen_class")[["DR_RF_all_argmax", "DR_RF_hmooa_gate"]])
s = show("drift_stream.csv", "Adaptive threshold under simulated drift")
st.line_chart(s.set_index("chunk")[["FPR_static", "FPR_adaptive"]])
st.line_chart(s.set_index("chunk")[["DR_static", "DR_adaptive"]])
