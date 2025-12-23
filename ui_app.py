import streamlit as st
import os
import pandas as pd
from video_metadata_agent import main as run_agent

st.set_page_config(page_title="Video Audit Agent", layout="centered")

st.title("🎥 Video Metadata Audit Agent")

# 🔐 API Key Input
api_key = st.text_input(
    "Paste OpenAI API Key",
    type="password",
    help="Key is used only for this session and not stored"
)

uploaded_file = st.file_uploader(
    "Upload metadata video (20–40 sec)",
    type=["mp4", "mov", "avi"]
)

if api_key and uploaded_file:
    os.makedirs("input_videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("input_videos", "sample.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded")

    if st.button("Run Audit"):
        with st.spinner("Processing video..."):
            run_agent(api_key)

        output_csv = "outputs/audit_metadata_output.csv"

        if os.path.exists(output_csv):
            # ✅ Load results
            df = pd.read_csv(output_csv)

            st.success("Audit completed")

            # ✅ SHOW RESULTS IN UI
            st.subheader("📊 Audit Results (Vertical View)")

            # Convert single-row CSV to vertical table
            vertical_df = df.T.reset_index()
            vertical_df.columns = ["Field", "Value"]

            st.dataframe(vertical_df, use_container_width=True)

            risk_level = df["risk_level"].iloc[0]
            risk_flags = df["risk_flags"].iloc[0]

            st.metric("🚨 Risk Level", risk_level)

        


            if isinstance(risk_flags, str) and risk_flags.strip():
                st.warning("⚠️ " + risk_flags)
            else:
                st.success("✅ No risk flags detected")


            # ✅ DOWNLOAD BUTTON
            st.download_button(
                label="⬇ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="audit_metadata_output.csv",
                mime="text/csv"
            )
        else:
            st.error("Audit completed but output CSV was not found.")
else:
    st.info("Paste API key and upload video to continue")
