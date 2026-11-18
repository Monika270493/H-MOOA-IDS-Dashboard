"""H-MOOA IDS - interactive results dashboard.
Run:  streamlit run dashboard_app.py
Click any bar to see its details."""
import os
import altair as alt, pandas as pd, streamlit as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
st.set_page_config(page_title="H-MOOA IDS Dashboard", page_icon="🛡️", layout="wide")

@st.cache_data
def load(f): return pd.read_csv(os.path.join(R, f))

fr, opt, wil = load("final_results.csv"), load("optimizer_comparison.csv"), load("optimizer_wilcoxon.csv")
cv, zd, dr = load("cv_statistics.csv"), load("zero_day.csv"), load("drift_stream.csv")
TEAL, GREY, ORANGE = "#0f6e7a", "#a9bcc3", "#c4571a"

def bar(df, cat, val, key, horizontal=True, height=320):
    """Bar chart where a click selects a bar. Returns the clicked category or None."""
    sel = alt.selection_point(name="sel", fields=[cat])
    q, n = alt.X(f"{val}:Q", title=val), alt.Y(f"{cat}:N", sort="-x", title=None)
    if not horizontal:
        q, n = alt.Y(f"{val}:Q", title=val), alt.X(f"{cat}:N", title=cat)
    chart = (alt.Chart(df).mark_bar().encode(
        q, n, color=alt.condition(sel, alt.value(TEAL), alt.value(GREY)), tooltip=[cat, val])
        .add_params(sel).properties(height=height))
    ev = st.altair_chart(chart, use_container_width=True, on_select="rerun", key=key)
    s = ev.selection.get("sel") if ev else None
    return s[0].get(cat) if s else None

st.title("🛡️ H-MOOA intrusion detection for IIoT")
st.caption("CIC-IIoT-2025 | 685,671 flows | 8 classes | click any bar for details")

best = fr.loc[fr["Accuracy"].idxmax()]
fus = fr[fr["Model"].str.contains("fusion")].iloc[0]
k = st.columns(4)
k[0].metric("Flows", "685,671"); k[1].metric("Features", "78 → 15", "-81%", delta_color="off")
k[2].metric("Best accuracy", f"{best.Accuracy:.2%}", best.Model, delta_color="off")
k[3].metric("Lowest benign false alarms", f"{fus.FPR_benign:.2%}", "Fusion model", delta_color="off")

t1, t2, t3, t4, t5 = st.tabs(["Models", "Optimizers", "Unseen attacks", "Drift", "Deployment"])

with t1:
    metric = st.selectbox("Metric", ["Accuracy", "F1", "FPR_benign", "Detection_rate", "Train_time_s", "N_features"], index=1)
    left, right = st.columns([3, 2])
    with left:
        pick = bar(fr, "Model", metric, "m_bar", height=480)
    with right:
        row = fr[fr["Model"] == (pick or best.Model)].iloc[0]
        st.subheader(row.Model)
        if not pick: st.caption("Click a bar to change the model. Showing the most accurate one.")
        a, b = st.columns(2)
        a.metric("Accuracy", f"{row.Accuracy:.2%}"); b.metric("Macro F1", f"{row.F1:.4f}")
        a.metric("Benign false alarms", f"{row.FPR_benign:.2%}"); b.metric("Attack detection", f"{row.Detection_rate:.2%}")
        a.metric("Features", int(row.N_features)); b.metric("Training time", f"{row.Train_time_s:.0f} s")
    with st.expander("Full table"):
        st.dataframe(fr, use_container_width=True)
    st.subheader("Repeated cross-validation (5 x 2 folds)")
    st.dataframe(cv, use_container_width=True)

with t2:
    om = st.radio("Compare optimizers by", ["HV_mean", "test_F1", "test_FPR_benign", "test_DetectionRate", "n_features"], horizontal=True)
    left, right = st.columns([3, 2])
    with left:
        pick = bar(opt, "Optimizer", om, "o_bar", horizontal=False)
    with right:
        name = pick or "H-MOOA"; r = opt[opt.Optimizer == name].iloc[0]
        st.subheader(name)
        a, b = st.columns(2)
        a.metric("Hypervolume", f"{r.HV_mean:.4f}"); b.metric("Test F1", f"{r.test_F1:.4f}")
        a.metric("Benign false alarms", f"{r.test_FPR_benign:.2%}"); b.metric("Features", f"{r.n_features:.1f}")
        w = wil[wil.competitor == name]
        if name == "H-MOOA": st.success("The proposed method. Compare it with the others by clicking their bars.")
        elif not w.empty:
            w = w.iloc[0]
            (st.success if w["significant (p_Holm<0.05)"] else st.warning)(
                f"H-MOOA vs {name}: Holm p = {w.p_Holm:.4f}. " + ("H-MOOA is significantly better." if w["significant (p_Holm<0.05)"] else "Not significant at 0.05."))
    st.dataframe(opt, use_container_width=True)

with t3:
    zd2 = zd.assign(unseen_class=zd.unseen_class.astype(str)).melt("unseen_class", ["DR_RF_all_argmax", "DR_RF_hmooa_gate"], "Method", "Detection rate")
    sel = alt.selection_point(name="sel", fields=["unseen_class"])
    ch = (alt.Chart(zd2).mark_bar().encode(
        x=alt.X("unseen_class:N", title="Class left out of training"), y=alt.Y("Detection rate:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset="Method:N", color=alt.Color("Method:N", scale=alt.Scale(range=[GREY, TEAL])),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.45)), tooltip=["unseen_class", "Method", "Detection rate"])
        .add_params(sel).properties(height=340))
    ev = st.altair_chart(ch, use_container_width=True, on_select="rerun", key="z_bar")
    s = ev.selection.get("sel") if ev else None
    c = s[0]["unseen_class"] if s else "7"
    z = zd[zd.unseen_class.astype(str) == c].iloc[0]
    st.subheader(f"Class {c} unseen")
    a, b, d = st.columns(3)
    a.metric("Plain Random Forest", f"{z.DR_RF_all_argmax:.1%}")
    b.metric("H-MOOA gate", f"{z.DR_RF_hmooa_gate:.1%}", f"{(z.DR_RF_hmooa_gate - z.DR_RF_all_argmax) * 100:+.1f} points")
    d.metric("Gate benign false alarms", f"{z.FPR_benign_gate:.2%}")
    if not s: st.caption("Click a class to see its details. Class 7 shown, the biggest improvement.")

with t4:
    ck = st.slider("Stream chunk (drift starts at chunk 10)", 0, 19, 12)
    r = dr[dr.chunk == ck].iloc[0]
    a, b, c3, d = st.columns(4)
    a.metric("False alarms, static τ", f"{r.FPR_static:.2%}"); b.metric("False alarms, adaptive τ", f"{r.FPR_adaptive:.2%}")
    c3.metric("Detection, static τ", f"{r.DR_static:.1%}"); d.metric("Detection, adaptive τ", f"{r.DR_adaptive:.1%}")
    for cols, title in ((["FPR_static", "FPR_adaptive"], "Benign false-alarm rate"), (["DR_static", "DR_adaptive"], "Attack detection rate")):
        m = dr.melt("chunk", cols, "Threshold", "Rate")
        line = alt.Chart(m).mark_line(point=True).encode(x="chunk:Q", y="Rate:Q", color=alt.Color("Threshold:N", scale=alt.Scale(range=[ORANGE, TEAL])))
        rule = alt.Chart(pd.DataFrame({"chunk": [ck]})).mark_rule(strokeDash=[4, 4]).encode(x="chunk:Q")
        st.subheader(title); st.altair_chart((line + rule).properties(height=260), use_container_width=True)
    st.info("The drift is simulated. The adaptive threshold lowers false alarms, but detection falls too, so the model still needs retraining.")

with t5:
    st.markdown("""
**Pipeline:** clean → encode text columns → MinMax scale → keep the 15 H-MOOA features → DT + RF (+ LSTM) fusion → alert if P(benign) < τ.

**Deploy this dashboard**
1. Upload `dashboard_app.py`, `requirements.txt`, `.streamlit/` and the `results/` folder to a public GitHub repository.
2. On share.streamlit.io choose the repository, set the main file to `dashboard_app.py`, pick Python 3.12, and click Deploy.

The live detector (upload a CSV, get alerts) needs the trained model files from the notebook's export cell.
""")
