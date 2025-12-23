import streamlit as st
import os
import pandas as pd
from video_metadata_agent import run_agent

st.set_page_config(page_title="Video Audit Agent", layout="centered")

st.title("🎥 Video Metadata Audit Agent")

# 🔐 API Key Input
api_key = st.text_input(
    "Paste OpenAI API Key",
    type="password",
    help="Key is used only for this session and not stored"
)

# 📤 Multiple video upload
uploaded_files = st.file_uploader(
    "Upload metadata videos (20–90 sec each)",
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

if api_key and uploaded_files:
    os.makedirs("input_videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_paths = []

    # Save uploaded videos
    for uploaded_file in uploaded_files:
        video_path = os.path.join("input_videos", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        video_paths.append(video_path)

    st.success(f"{len(video_paths)} video(s) uploaded")

    if st.button("Run Audit"):
        with st.spinner("Processing video(s)..."):
            run_agent(api_key, video_paths)

        output_csv = "outputs/audit_metadata_output.csv"

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)

            st.success("Audit completed")
            st.subheader("📊 Audit Results (All Videos)")

            # 🔁 SHOW ALL RESULTS ONE AFTER ANOTHER
            for idx, row in df.iterrows():
                st.markdown(f"### 🎬 Video {idx + 1}")

                if "video_name_value" in row and pd.notna(row["video_name_value"]):
                    st.caption(f"File: {row['video_name_value']}")

                record = row.to_dict()

                vertical_df = pd.DataFrame(
                    list(record.items()),
                    columns=["Field", "Value"]
                )

                st.dataframe(vertical_df, use_container_width=True)

                # ---- RISK DISPLAY ----
                risk_level = record.get("risk_level")
                risk_flags = record.get("risk_flags")

                if risk_level:
                    st.metric("🚨 Risk Level", risk_level)

                if isinstance(risk_flags, str) and risk_flags.strip():
                    st.warning("⚠️ " + risk_flags)
                else:
                    st.success("✅ No risk flags detected")

                st.markdown("---")  # separator between videos

            # ---- DOWNLOAD ----
            st.download_button(
                label="⬇ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="audit_metadata_output.csv",
                mime="text/csv"
            )
        else:
            st.error("Audit completed but output CSV was not found.")
else:
    st.info("Paste API key and upload one or more videos to continue")
