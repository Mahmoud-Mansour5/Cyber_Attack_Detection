import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Intrusion Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0f1525 !important; border-right: 1px solid #1e2d4a; }

.header-banner {
    background: linear-gradient(135deg, #0f2027, #1a3a5c, #0f2027);
    border: 1px solid #1e4d7a; border-radius: 12px;
    padding: 28px 36px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(0,200,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.header-title {
    font-family: 'JetBrains Mono', monospace; font-size: 1.9rem;
    font-weight: 700; color: #00d4ff; letter-spacing: -0.5px; margin: 0 0 6px 0;
}
.header-sub {
    font-size: 0.88rem; color: #7a9bbf; font-weight: 400;
    margin: 0; font-family: 'JetBrains Mono', monospace;
}
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
.metric-card { background: #111827; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px; text-align: center; }
.metric-label { font-size: 0.72rem; color: #4a6fa5; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #00d4ff; }

.section-header {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #4a6fa5;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 24px 0 12px 0; border-bottom: 1px solid #1e2d4a; padding-bottom: 8px;
}
.input-group-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #00d4ff;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;
}
.result-attack {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 2px solid #dc2626; border-radius: 14px;
    padding: 28px 36px; text-align: center; margin: 20px 0;
}
.result-normal {
    background: linear-gradient(135deg, #0a1a0f, #0f2d18);
    border: 2px solid #16a34a; border-radius: 14px;
    padding: 28px 36px; text-align: center; margin: 20px 0;
}
.result-icon  { font-size: 3rem; margin-bottom: 10px; }
.result-label { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }
.result-desc  { font-size: 0.9rem; color: #94a3b8; }

.sample-card { background: #0f1c2e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
.sample-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
.tag-normal { background: #14532d; color: #4ade80; }
.tag-attack { background: #450a0a; color: #f87171; }
.tag-random { background: #1e3a5f; color: #60a5fa; }
.divider { border: none; border-top: 1px solid #1e2d4a; margin: 20px 0; }
.score-bar-wrap { background: #111827; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px; margin-top: 12px; }
.score-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #4a6fa5; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.score-bar-track { background: #1e2d4a; border-radius: 6px; height: 10px; overflow: hidden; }
.score-bar-fill-attack { background: linear-gradient(90deg, #dc2626, #f87171); height: 100%; border-radius: 6px; }
.score-bar-fill-normal { background: linear-gradient(90deg, #16a34a, #4ade80); height: 100%; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (defined BEFORE use)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    pipeline = joblib.load('model_artifacts/svm_intrusion_pipeline.pkl')
    meta     = joblib.load('model_artifacts/pipeline_metadata.pkl')
    return pipeline, meta


@st.cache_data
def load_test_data():
    """Reproduce the same 80/20 stratified split used during training."""
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    from sklearn.model_selection import train_test_split
    X          = df.drop(columns=['id', 'label', 'attack_cat'], errors='ignore')
    y          = df['label']
    attack_cat = df['attack_cat'] if 'attack_cat' in df.columns else y.copy()
    _, X_test, _, y_test, _, cat_test = train_test_split(
        X, y, attack_cat, test_size=0.20, random_state=42, stratify=y
    )
    X_test = X_test.copy()
    X_test['__label__']      = y_test.values
    X_test['__attack_cat__'] = cat_test.values
    return X_test


def show_result(prediction, score, confidence, input_df):
    """Render the shared prediction result block."""
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    res_col, score_col = st.columns([3, 2], gap="large")

    with res_col:
        if prediction == 1:
            st.markdown("""
            <div class="result-attack">
                <div class="result-icon">🚨</div>
                <div class="result-label" style="color:#f87171;">ATTACK DETECTED</div>
                <div class="result-desc">This network flow has been classified as <b>malicious / attack traffic</b>.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <div class="result-icon">✅</div>
                <div class="result-label" style="color:#4ade80;">NORMAL TRAFFIC</div>
                <div class="result-desc">This network flow has been classified as <b>benign / normal traffic</b>.</div>
            </div>""", unsafe_allow_html=True)

    with score_col:
        fill      = confidence * 100
        bar_class = "score-bar-fill-attack" if prediction == 1 else "score-bar-fill-normal"
        bar_color = "#f87171" if prediction == 1 else "#4ade80"
        st.markdown(f"""
        <div class="score-bar-wrap">
            <div class="score-bar-label">Confidence Score</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:{bar_color};margin-bottom:10px;">{confidence:.2%}</div>
            <div class="score-bar-track"><div class="{bar_class}" style="width:{fill:.1f}%;"></div></div>
        </div>
        <div class="score-bar-wrap">
            <div class="score-bar-label">Decision Function Score</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600;color:#7a9bbf;">{score:.4f}</div>
            <div style="font-size:0.75rem;color:#4a6fa5;margin-top:4px;">Positive → Attack &nbsp;|&nbsp; Negative → Normal</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📋 View Input Features"):
        st.dataframe(
            input_df.T.rename(columns={input_df.index[0]: 'Value'}),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL & RENDER PAGE
# ══════════════════════════════════════════════════════════════════════════════
pipeline, meta = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-title">🛡️ Network Intrusion Detection System</div>
    <p class="header-sub">UNSW-NB15 Dataset &nbsp;·&nbsp; LinearSVC + PCA Pipeline &nbsp;·&nbsp; Alexandria National University &nbsp;·&nbsp; Data Computation Spring '26</p>
</div>
""", unsafe_allow_html=True)

# ─── Metrics Row ──────────────────────────────────────────────────────────────
acc   = meta['metrics']['accuracy']
f1    = meta['metrics']['f1']
prec  = meta['metrics']['precision']
rec   = meta['metrics']['recall']
n_pca = meta['n_pca']

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1rem;">LinearSVC</div></div>
    <div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{acc:.2%}</div></div>
    <div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-value">{f1:.2%}</div></div>
    <div class="metric-card"><div class="metric-label">PCA Dims</div><div class="metric-value">{n_pca}</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — MODE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;
     color:#00d4ff;padding:10px 0 16px 0;letter-spacing:-0.3px;">⚙️ Detection Mode</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    label="",
    options=["✏️  Manual Input", "🗄️  Sample from Test Data"],
    index=0
)
st.sidebar.markdown('<hr style="border-color:#1e2d4a;margin:10px 0 18px 0">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — MANUAL INPUT
# ══════════════════════════════════════════════════════════════════════════════
if mode == "✏️  Manual Input":

    st.sidebar.markdown(
        '<div style="font-size:0.8rem;color:#4a6fa5;font-family:\'JetBrains Mono\',monospace;">'
        'Enter feature values in the main area,<br>then press Predict below.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Manual Traffic Feature Input</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="input-group-title">🔌 Protocol & Connection</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        proto   = c1.selectbox("proto",   ["tcp","udp","arp","ospf","icmp","igmp","other"])
        service = c2.selectbox("service", ["-","ftp","smtp","ssh","dns","http","irc","pop3","snmp","ssl","dhcp"])
        state   = c3.selectbox("state",   ["INT","FIN","CON","REQ","RST","URN","PAR","ECO","ECR","URH"])

        st.markdown("")
        st.markdown('<div class="input-group-title">📦 Traffic Volume</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        dur  = c1.number_input("Duration (dur)", min_value=0.0, value=0.001, format="%.6f")
        rate = c2.number_input("Rate (pps)",      min_value=0.0, value=1000.0)
        c1, c2, c3, c4 = st.columns(4)
        spkts  = c1.number_input("spkts",  min_value=0, value=5)
        dpkts  = c2.number_input("dpkts",  min_value=0, value=3)
        sbytes = c3.number_input("sbytes", min_value=0, value=500)
        dbytes = c4.number_input("dbytes", min_value=0, value=300)
        c1, c2, c3, c4 = st.columns(4)
        sloss = c1.number_input("sloss", min_value=0, value=0)
        dloss = c2.number_input("dloss", min_value=0, value=0)
        smean = c3.number_input("smean", min_value=0, value=100)
        dmean = c4.number_input("dmean", min_value=0, value=100)

        st.markdown("")
        st.markdown('<div class="input-group-title">📡 Load & Jitter</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        sload = c1.number_input("sload", min_value=0.0, value=1000.0)
        dload = c2.number_input("dload", min_value=0.0, value=500.0)
        c1, c2, c3, c4 = st.columns(4)
        sinpkt = c1.number_input("sinpkt", min_value=0.0, value=0.0)
        dinpkt = c2.number_input("dinpkt", min_value=0.0, value=0.0)
        sjit   = c3.number_input("sjit",   min_value=0.0, value=0.0)
        djit   = c4.number_input("djit",   min_value=0.0, value=0.0)

    with col_right:
        st.markdown('<div class="input-group-title">🕐 TTL & TCP Timing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        sttl = c1.slider("sttl", 0, 255, 64)
        dttl = c2.slider("dttl", 0, 255, 64)
        c1, c2, c3 = st.columns(3)
        tcprtt = c1.number_input("tcprtt", min_value=0.0, value=0.0)
        synack = c2.number_input("synack", min_value=0.0, value=0.0)
        ackdat = c3.number_input("ackdat", min_value=0.0, value=0.0)
        c1, c2, c3, c4 = st.columns(4)
        swin  = c1.number_input("swin",  min_value=0, value=255)
        dwin  = c2.number_input("dwin",  min_value=0, value=255)
        stcpb = c3.number_input("stcpb", min_value=0, value=0)
        dtcpb = c4.number_input("dtcpb", min_value=0, value=0)

        st.markdown("")
        st.markdown('<div class="input-group-title">🔗 Connection Count Features (ct_*)</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        ct_srv_src       = c1.slider("ct_srv_src",       0, 100, 2)
        ct_state_ttl     = c2.slider("ct_state_ttl",     0, 10,  2)
        ct_dst_ltm       = c3.slider("ct_dst_ltm",       0, 100, 1)
        ct_src_dport_ltm = c4.slider("ct_src_dport_ltm", 0, 100, 1)
        c1, c2, c3, c4 = st.columns(4)
        ct_dst_sport_ltm = c1.slider("ct_dst_sport_ltm", 0, 100, 1)
        ct_dst_src_ltm   = c2.slider("ct_dst_src_ltm",   0, 100, 2)
        ct_src_ltm       = c3.slider("ct_src_ltm",       0, 100, 1)
        ct_srv_dst       = c4.slider("ct_srv_dst",       0, 100, 2)

        st.markdown("")
        st.markdown('<div class="input-group-title">🌐 Application & Flags</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        trans_depth       = c1.number_input("trans_depth",       min_value=0, value=0)
        response_body_len = c2.number_input("response_body_len", min_value=0, value=0)
        ct_ftp_cmd        = c3.number_input("ct_ftp_cmd",        min_value=0, value=0)
        c1, c2, c3 = st.columns(3)
        ct_flw_http_mthd = c1.number_input("ct_flw_http_mthd", min_value=0, value=0)
        is_ftp_login     = c2.selectbox("is_ftp_login",    [0, 1])
        is_sm_ips_ports  = c3.selectbox("is_sm_ips_ports", [0, 1])

    st.markdown("")
    predict_btn = st.button("🔍  Predict Traffic Class", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            "dur": dur, "proto": proto, "service": service, "state": state,
            "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
            "rate": rate, "sttl": sttl, "dttl": dttl, "sload": sload, "dload": dload,
            "sloss": sloss, "dloss": dloss, "sinpkt": sinpkt, "dinpkt": dinpkt,
            "sjit": sjit, "djit": djit, "swin": swin, "stcpb": stcpb,
            "dtcpb": dtcpb, "dwin": dwin, "tcprtt": tcprtt, "synack": synack,
            "ackdat": ackdat, "smean": smean, "dmean": dmean,
            "trans_depth": trans_depth, "response_body_len": response_body_len,
            "ct_srv_src": ct_srv_src, "ct_state_ttl": ct_state_ttl,
            "ct_dst_ltm": ct_dst_ltm, "ct_src_dport_ltm": ct_src_dport_ltm,
            "ct_dst_sport_ltm": ct_dst_sport_ltm, "ct_dst_src_ltm": ct_dst_src_ltm,
            "is_ftp_login": is_ftp_login, "ct_ftp_cmd": ct_ftp_cmd,
            "ct_flw_http_mthd": ct_flw_http_mthd, "ct_src_ltm": ct_src_ltm,
            "ct_srv_dst": ct_srv_dst, "is_sm_ips_ports": is_sm_ips_ports
        }
        input_df = pd.DataFrame([input_dict])[meta["feature_columns"]]
        prediction = pipeline.predict(input_df)[0]
        score      = pipeline.decision_function(input_df)[0]
        confidence = 1 / (1 + np.exp(-abs(score)))
        show_result(prediction, score, confidence, input_df)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — SAMPLE FROM TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#4a6fa5;'
        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Sample Type</div>',
        unsafe_allow_html=True
    )
    sample_type = st.sidebar.radio(
        label="",
        options=["🎲  Random Sample", "✅  Random Normal Traffic", "🚨  Random Attack Traffic"],
        index=0
    )
    st.sidebar.markdown('<hr style="border-color:#1e2d4a;margin:14px 0">', unsafe_allow_html=True)
    fetch_btn = st.sidebar.button("⚡  Fetch & Predict", type="primary", use_container_width=True)

    st.markdown('<div class="section-header">Test Set Sampling</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:10px;
         padding:16px 20px;margin-bottom:20px;font-size:0.86rem;color:#7a9bbf;">
        <b style="color:#00d4ff;">How it works:</b> Picks a random record from the 20% held-out test split
        (same split used during model evaluation, <code>random_state=42</code>).
        The true label is revealed <em>after</em> prediction so you can verify correctness.
    </div>
    """, unsafe_allow_html=True)

    if fetch_btn:
        try:
            test_df = load_test_data()
        except Exception as e:
            st.error(f"❌ Could not load dataset: {e}")
            st.stop()

        label_col = '__label__'
        if sample_type == "✅  Random Normal Traffic":
            pool     = test_df[test_df[label_col] == 0]
            tag_html = '<span class="sample-tag tag-normal">NORMAL</span>'
        elif sample_type == "🚨  Random Attack Traffic":
            pool     = test_df[test_df[label_col] == 1]
            tag_html = '<span class="sample-tag tag-attack">ATTACK</span>'
        else:
            pool     = test_df
            tag_html = '<span class="sample-tag tag-random">RANDOM</span>'

        sample_row = pool.sample(1, random_state=None)
        true_label = int(sample_row[label_col].values[0])
        attack_cat = sample_row['__attack_cat__'].values[0] if '__attack_cat__' in sample_row.columns else "N/A"

        feature_df = sample_row.drop(
            columns=[c for c in ['__label__', '__attack_cat__'] if c in sample_row.columns]
        )
        feature_df = feature_df[meta["feature_columns"]]

        prediction = pipeline.predict(feature_df)[0]
        score      = pipeline.decision_function(feature_df)[0]
        confidence = 1 / (1 + np.exp(-abs(score)))

        true_label_str = "Normal" if true_label == 0 else "Attack"
        true_color     = "#4ade80" if true_label == 0 else "#f87171"
        correct        = (prediction == true_label)
        verdict        = "✅ Correct" if correct else "❌ Wrong"
        verdict_color  = "#4ade80" if correct else "#f87171"

        st.markdown(f"""
        <div class="sample-card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#4a6fa5;">SAMPLE INFO</span>
                {tag_html}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;font-size:0.85rem;">
                <div>
                    <div style="color:#4a6fa5;font-size:0.72rem;font-family:'JetBrains Mono',monospace;margin-bottom:3px;">TRUE LABEL</div>
                    <div style="color:{true_color};font-weight:600;">{true_label_str}</div>
                </div>
                <div>
                    <div style="color:#4a6fa5;font-size:0.72rem;font-family:'JetBrains Mono',monospace;margin-bottom:3px;">ATTACK CATEGORY</div>
                    <div style="color:#e2e8f0;">{attack_cat}</div>
                </div>
                <div>
                    <div style="color:#4a6fa5;font-size:0.72rem;font-family:'JetBrains Mono',monospace;margin-bottom:3px;">VERDICT</div>
                    <div style="color:{verdict_color};font-weight:600;">{verdict}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        show_result(prediction, score, confidence, feature_df)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.35;">🗄️</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#2a4a6a;">
                Press <b style="color:#00d4ff;">Fetch &amp; Predict</b> in the sidebar<br>to load a random test sample.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'JetBrains Mono',monospace;
     font-size:0.72rem;color:#2a4a6a;padding:8px 0 16px 0;">
    Alexandria National University &nbsp;·&nbsp; Data Computation Spring '26
    &nbsp;·&nbsp; UNSW-NB15 SVM Intrusion Detection System
</div>
""", unsafe_allow_html=True)