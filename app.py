import csv
import time
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="SilentGuard AI - Phase 3",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_data
def load_class_names():
    class_map_path = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    )
    with open(class_map_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["display_name"] for row in rows]

def prepare_audio(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_path = Path("._silentguard_input" + (suffix if suffix else ".wav"))
    temp_path.write_bytes(uploaded_file.getbuffer())
    try:
        waveform, _ = librosa.load(temp_path, sr=16000, mono=True)
    finally:
        temp_path.unlink(missing_ok=True)
    return waveform.astype(np.float32)

def classify_audio(waveform):
    model = load_yamnet()
    class_names = load_class_names()
    scores, _, _ = model(waveform)
    mean_scores = scores.numpy().mean(axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:5]
    return [(class_names[int(i)], float(mean_scores[int(i)])) for i in top_indices]

# Phase 2: SilentGuard-specific interpretation of generic YAMNet labels.
EVENT_RULES = [
    (("glass", "shatter", "breaking glass", "crack"), "🔨 Glass Breaking", "HIGH"),
    (("scream", "shout", "yell", "crying"), "🗣️ Distress / Raised Voice", "CRITICAL"),
    (("impact", "thump", "thud", "slam", "bang", "crash"), "💥 Impact / Fall-like Sound", "HIGH"),
    (("water", "pour", "tap", "faucet", "splash"), "💧 Water-related Sound", "MEDIUM"),
]

def map_to_silentguard(top_predictions):
    label = top_predictions[0][0].lower()
    for keywords, event, risk in EVENT_RULES:
        if any(keyword in label for keyword in keywords):
            return event, risk, f'YAMNet top label "{top_predictions[0][0]}" matched a SilentGuard prototype category.'
    return "🟢 Normal / Unclassified Environment", "LOW", (
        f'YAMNet top label "{top_predictions[0][0]}" did not match a configured '
        "emergency-oriented category."
    )

# ---------------------------
# Session state
# ---------------------------
DEFAULTS = {
    "last_analysis": None,
    "verification_status": None,
    "verification_deadline": None,
    "uploaded_name": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("🛡️ SilentGuard AI")
st.subheader("Sound Detection, Risk & Emergency Verification")
st.info(
    "Prototype flow: audio → YAMNet → SilentGuard risk → 30-second verification → simulated emergency response."
)

uploaded_file = st.file_uploader(
    "🎤 Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Sound", type="primary", use_container_width=True):
        st.session_state.verification_status = None
        st.session_state.verification_deadline = None
        st.session_state.last_analysis = None
        st.session_state.uploaded_name = uploaded_file.name

        with st.spinner("Loading YAMNet and analyzing audio..."):
            try:
                waveform = prepare_audio(uploaded_file)
                results = classify_audio(waveform)
                top_label, top_score = results[0]
                event, risk, reason = map_to_silentguard(results)

                st.session_state.last_analysis = {
                    "results": results,
                    "event": event,
                    "risk": risk,
                    "label": top_label,
                    "confidence": top_score,
                    "reason": reason,
                }

                # Start the 30-second response window only for HIGH/CRITICAL risk.
                if risk in ("HIGH", "CRITICAL"):
                    st.session_state.verification_deadline = time.time() + 30

                st.rerun()
            except Exception as exc:
                st.error("The audio could not be analyzed.")
                st.code(str(exc))
                st.info(
                    "If this is the first run, check your internet connection because "
                    "the pretrained YAMNet model and class map must be downloaded."
                )

# ---------------------------
# Display saved analysis
# ---------------------------
analysis = st.session_state.last_analysis

if analysis is not None:
    results = analysis["results"]
    risk = analysis["risk"]

    st.success("Analysis completed")

    st.markdown("### 🤖 YAMNet Result")
    st.metric(
        "Detected Sound",
        analysis["label"],
        f"{analysis['confidence'] * 100:.2f}% confidence",
    )

    st.markdown("### 🛡️ SilentGuard Interpretation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Event Category", analysis["event"])
    with col2:
        st.metric("Prototype Risk", risk)

    if risk == "CRITICAL":
        st.error("🔴 CRITICAL RISK")
    elif risk == "HIGH":
        st.warning("🔴 HIGH RISK")
    elif risk == "MEDIUM":
        st.info("🟡 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

    st.caption(f"Why: {analysis['reason']}")

    # ---------------------------
    # Phase 3 interactive verification
    # ---------------------------
    st.markdown("---")
    st.markdown("### 🗣️ Emergency Verification")

    if risk in ("HIGH", "CRITICAL"):
        # Refresh once per second so the 30-second countdown is automatic.
        if st.session_state.verification_status is None and st.session_state.verification_deadline:
            st_autorefresh(interval=1000, key="verification_timer")

        if st.session_state.verification_status is None:
            remaining = max(0, int(np.ceil(st.session_state.verification_deadline - time.time())))

            if remaining <= 0:
                st.session_state.verification_status = "NO_RESPONSE"
                st.session_state.verification_deadline = None
                st.rerun()

            st.warning(
                "A high-risk sound classification was detected. Please confirm your safety."
            )
            st.markdown("#### Are you okay?")
            st.progress(remaining / 30.0, text=f"⏱️ Response window: {remaining} seconds")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🟢 I'm OK", use_container_width=True, key="ok_button"):
                    st.session_state.verification_status = "OK"
                    st.session_state.verification_deadline = None
                    st.rerun()
            with c2:
                if st.button("🔴 I Need Help", use_container_width=True, key="help_button"):
                    st.session_state.verification_status = "HELP"
                    st.session_state.verification_deadline = None
                    st.rerun()

            st.caption(
                "If neither button is selected within 30 seconds, SilentGuard automatically starts the no-response escalation simulation."
            )

        status = st.session_state.verification_status

        if status == "OK":
            st.success("🟢 You're okay — fine, sorry for the interruption. Emergency escalation has been cancelled.")
            st.info("Monitoring mode restored. No alert was sent.")

        elif status == "HELP":
            st.error("🚨 HELP CONFIRMED")
            st.markdown("#### 📡 Emergency Response — Simulation")
            r1, r2 = st.columns(2)
            with r1:
                st.success("✅ Family message\n\n**SENT — SIMULATED**")
            with r2:
                st.warning("📞 Ambulance / emergency call\n\n**INITIATED — SIMULATED**")
            st.info("🏥 Nearby hospital notification\n\n**SHARED — SIMULATED**")
            st.caption(
                "Prototype only: no real SMS, phone call, hospital notification, ambulance dispatch, or external API request is made."
            )

        elif status == "NO_RESPONSE":
            st.error("🚨 NO RESPONSE — AUTOMATIC ESCALATION")
            st.markdown("#### 📡 Emergency Response — Simulation")
            r1, r2 = st.columns(2)
            with r1:
                st.success("✅ Family message\n\n**AUTO-SENT — SIMULATED**")
            with r2:
                st.warning("📞 Ambulance / emergency call\n\n**AUTO-INITIATED — SIMULATED**")
            st.info("🏥 Nearby hospital notification\n\n**AUTO-SHARED — SIMULATED**")
            st.caption(
                "The 30-second window expired without a response. This is a software-only demonstration; no real emergency service is contacted."
            )
    else:
        st.success("🟢 No emergency verification is triggered for the current LOW/MEDIUM prototype risk.")

    st.markdown("### 📊 Top 5 YAMNet Predictions")
    for rank, (label, score) in enumerate(results, start=1):
        st.write(f"**{rank}. {label}** — {score * 100:.2f}%")
        st.progress(float(min(max(score, 0.0), 1.0)))

    st.caption(
        "Risk labels and emergency actions are prototype rules for demonstration. "
        "No real emergency alert is sent."
    )
else:
    st.markdown(
        """
        **Workflow**

        `🎤 Audio → 🤖 YAMNet → 🛡️ Event/Risk → 🗣️ 30s Verification → 🚨 Alert Simulation`

        Upload a short environmental-audio sample and click **Analyze Sound** to begin.
        """
    )
