import os
import cv2
import json
import csv
import base64
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import re
import pandas as pd


# ---------------------------
# GLOBAL API COST TRACKING
# ---------------------------
VISION_CALL_COUNT = 0

# ---------------------------
# TESSERACT CONFIG
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------
# CONSTANTS
# ---------------------------
FRAME_INTERVAL_SECONDS = 2
MAX_FRAMES = 20

VIDEO_PATH = "input_videos/sample.mp4"
OUTPUT_JSON = "outputs/audit_metadata_output.json"
OUTPUT_CSV = "outputs/audit_metadata_output.csv"

# ---------------------------
# DEFAULT AUDIT SCHEMA
# ---------------------------
DEFAULT_SCHEMA = {
    "location": {"value": None, "visible": False},
    "date": {"value": None, "visible": False},
    "time": {"value": None, "visible": False},
    "device_model": {"value": None, "visible": False},
    "os_version": {"value": None, "visible": False},
    "network": {"value": None, "visible": False},
    "carrier": {"value": None, "visible": False},
    "map_provider": {"value": None, "visible": False},
    "vpn": {"value": False, "visible": False},
    "mock_location": {"value": False, "visible": False},
    "developer_options": {"value": False, "visible": False},
    "screen_recording": {"value": False, "visible": False},
    "session_id": {"value": None, "visible": False},
    "build_version": {"value": None, "visible": False}

}

# ---------------------------
# OPENAI CLIENT
# ---------------------------
def get_client(api_key: str):
    return OpenAI(api_key=api_key)

# ---------------------------
# VIDEO PROCESSING
# ---------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps * FRAME_INTERVAL_SECONDS)

    frames, fid = [], 0
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if fid % interval == 0:
            frames.append(frame)
        fid += 1

    cap.release()
    return frames


def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


# ---------------------------
# AI ANALYSIS
# ---------------------------
def analyze_frame(encoded, client):
    """
    Returns a list of visible text lines using:
    1. GPT-4o Vision (best effort)
    2. Tesseract OCR (guaranteed fallback)
    """

    text_lines = []

    # ---------- TRY GPT VISION ----------
    prompt = "List all readable text visible on this screenshot. Return plain text."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            }
                        }
                    ]
                }
            ]
        )

        content = response.choices[0].message.content

        if content:
            # Split lines safely
            text_lines.extend(
                [line.strip() for line in content.splitlines() if line.strip()]
            )

    except Exception:
        pass

    # ---------- OCR FALLBACK ----------
    try:
        image_bytes = base64.b64decode(encoded)
        img = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        ocr_text = pytesseract.image_to_string(img)

        if ocr_text:
            text_lines.extend(
                [line.strip() for line in ocr_text.splitlines() if line.strip()]
            )

    except Exception:
        pass

    return text_lines


def derive_fields(text_lines):
    """
    Convert raw visible text lines into structured audit fields
    with {value, visible} schema.
    """

    import re
    import json

    # Clone default schema
    data = json.loads(json.dumps(DEFAULT_SCHEMA))

    # Normalize all text
    combined_text = " ".join(text_lines).lower()

    # --------------------------------------------------
    # MAP PROVIDER + LOCATION (IMPORTANT FIX)
    # --------------------------------------------------

    # Detect map UI even if logo text is missing
    map_indicators = ["map", "maps", "gps", "navigation", "route", "compass"]

    if "google maps" in combined_text:
        data["map_provider"] = {
            "value": "Google Maps",
            "visible": True
        }
    elif any(word in combined_text for word in map_indicators):
        data["map_provider"] = {
            "value": "Map UI Visible",
            "visible": True
        }

    # Location name if readable
    location_patterns = [
        r"bengaluru",
        r"bangalore",
        r"chennai",
        r"mumbai",
        r"delhi",
        r"hyderabad",
        r"india"
    ]

    for loc in location_patterns:
        if loc in combined_text:
            data["location"] = {
                "value": loc.title() if loc != "india" else "India",
                "visible": True
            }
            break

    # Fallback: map visible ⇒ location visible (audit-correct)
    if not data["location"]["visible"] and data["map_provider"]["visible"]:
        data["location"] = {
            "value": "Location visible on map",
            "visible": True
        }

    # --------------------------------------------------
    # DATE + TIME (IMPORTANT FIX)
    # --------------------------------------------------

    # Multiple date formats
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}\s?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, combined_text)
        if match:
            data["date"] = {
                "value": match.group(),
                "visible": True
            }
            break

    # Time
    time_match = re.search(r"\b\d{1,2}:\d{2}(\s?(am|pm|ist))?\b", combined_text)
    if time_match:
        data["time"] = {
            "value": time_match.group(),
            "visible": True
        }

    # Fallback: time visible ⇒ date context visible
    if data["time"]["visible"] and not data["date"]["visible"]:
        data["date"] = {
            "value": "Visible (via timestamp)",
            "visible": True
        }

    # --------------------------------------------------
    # DEVICE MODEL (DYNAMIC)
    # --------------------------------------------------

   # ---------------- DEVICE MODEL (FINAL & CORRECT) ----------------

# Normalize OCR text to handle iphone11 / iphone-11 / iphone_11
    normalized_text = combined_text.replace("-", " ").replace("_", " ")

    # iPhone has highest priority
    iphone_match = re.search(r"\biphone\s*\d+\b", normalized_text)
    if iphone_match:
        data["device_model"] = {
            "value": iphone_match.group().title(),
            "visible": True
        }
    else:
    # Android devices ONLY if no iPhone found
        android_match = re.search(
            r"\b(pixel\s*\d+[a-z]*|samsung\s*galaxy\s*[a-z0-9]+|oneplus\s*\d+|redmi\s*\w+|realme\s*\w+)\b",
            normalized_text
        )
        if android_match:
            data["device_model"] = {
                "value": android_match.group().title(),
                "visible": True
            }



    # --------------------------------------------------
    # OS VERSION
    # --------------------------------------------------

    os_match = re.search(r"android\s*\d+", combined_text)
    if os_match:
        data["os_version"] = {
            "value": os_match.group().title(),
            "visible": True
        }

            # ---------------- SESSION ID ----------------
    session_match = re.search(
        r"(session\s*id|sessionid|sid)[:\s]*([a-zA-Z0-9\-]+)",
        combined_text
    )

    if session_match:
        data["session_id"] = {
            "value": session_match.group(2),
            "visible": True
        }


    # --------------------------------------------------
    # NETWORK TYPE
    # --------------------------------------------------

    network_patterns = {
        "Wi-Fi": r"wi[\s-]?fi",
        "5G": r"\b5g\b",
        "4G": r"\b4g\b",
        "LTE": r"\blte\b"
    }

    for name, pattern in network_patterns.items():
        if re.search(pattern, combined_text):
            data["network"] = {
                "value": name,
                "visible": True
            }
            break

    # --------------------------------------------------
    # CARRIER
    # --------------------------------------------------

    carrier_patterns = [
        "airtel", "jio", "vodafone", "vi", "idea",
        "verizon", "att", "t-mobile"
    ]

    for carrier in carrier_patterns:
        if carrier in combined_text:
            data["carrier"] = {
                "value": carrier.upper() if carrier == "vi" else carrier.title(),
                "visible": True
            }
            break

    # --------------------------------------------------
    # VPN (CONTEXT-AWARE)
    # --------------------------------------------------

    if "vpn" in combined_text:
        if "not visible" in combined_text or "disabled" in combined_text:
            data["vpn"] = {"value": False, "visible": True}
        else:
            data["vpn"] = {"value": True, "visible": True}

    # --------------------------------------------------
    # MOCK LOCATION
    # --------------------------------------------------

    if "mock location" in combined_text:
        if "not visible" in combined_text or "disabled" in combined_text:
            data["mock_location"] = {"value": False, "visible": True}
        else:
            data["mock_location"] = {"value": True, "visible": True}

    # --------------------------------------------------
    # DEVELOPER OPTIONS
    # --------------------------------------------------

    if "developer options" in combined_text:
        if "enabled" in combined_text:
            data["developer_options"] = {"value": True, "visible": True}
        else:
            data["developer_options"] = {"value": False, "visible": True}

         # ---------------- BUILD VERSION ----------------
    # ---------------- BUILD VERSION (FLEXIBLE: 2–20 SEGMENTS) ----------------
    build_match = re.search(r"\b\d+(?:\.\d+){1,19}\b", combined_text)
    if build_match:
        data["build_version"] = {
            "value": build_match.group(),
            "visible": True
        }

        

    # --------------------------------------------------
    # SCREEN RECORDING
    # --------------------------------------------------

    if "screen recording" in combined_text:
        data["screen_recording"] = {"value": True, "visible": True}

    return data

def analyze_frame_semantic(encoded, client):
    """
    One-time GPT Vision call to extract semantic metadata
    """
    global VISION_CALL_COUNT
    VISION_CALL_COUNT += 1

    prompt = """
Extract ONLY the following if clearly visible.
Do not guess.

Return JSON:
{
  location,
  date,
  time,
  device_model,
  os_version,
  map_provider
}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]
        }]
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {}
    
   



# ---------------------------
# NORMALIZATION
# ---------------------------
def normalize_schema(raw_results):
    normalized = json.loads(json.dumps(DEFAULT_SCHEMA))

    for result in raw_results:
        for key, obj in result.items():
            if key in normalized and isinstance(obj, dict):
                if obj.get("visible") is True:
                    normalized[key] = {
                        "value": obj.get("value"),
                        "visible": True
                    }

    return normalized




# ---------------------------
# RISK LOGIC
# ---------------------------
def calculate_risk(data):
    flags = []

    if data["vpn"]["value"] is True:
        flags.append("VPN Detected")

    if data["mock_location"]["value"] is True:
        flags.append("Mock Location Detected")

    if data["developer_options"]["value"] is True:
        flags.append("Developer Options Enabled")

    if not data["location"]["visible"]:
        flags.append("Location Not Visible")

    if not data["date"]["visible"]:
        flags.append("Date Not Visible")

    if len(flags) >= 3:
        return "HIGH", flags
    elif len(flags) == 2:
        return "MEDIUM", flags
    else:
        return "LOW", flags
    


# ---------------------------
# CSV OUTPUT (FLATTENED)
# ---------------------------
def save_csv(data, risk, flags):
    flat = {}

    for key, obj in data.items():
        # Handle renamed field
        if key == "time":
            flat["time_captured"] = obj.get("value")
        else:
            flat[f"{key}_value"] = obj.get("value")

        # Keep visibility only where required
        if key in ["location", "date", "time", "map_provider"]:
            flat[f"{key}_visible"] = obj.get("visible")

    flat["risk_level"] = risk
    flat["risk_flags"] = "; ".join(flags)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)

def save_csv_multiple(results):
    rows = []

    for data in results:
        flat = {}

        for key, obj in data.items():
            if isinstance(obj, dict):
                flat[f"{key}_value"] = obj.get("value")

        flat["risk_level"] = data.get("_risk_level")
        flat["risk_flags"] = data.get("_risk_flags")

        rows.append(flat)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")




def analyze_frame_semantic(encoded, client):
    """
    One-time GPT Vision call for semantic metadata only
    """
    global VISION_CALL_COUNT
    VISION_CALL_COUNT += 1

    prompt = """
Extract ONLY if clearly visible on screen. Do not guess or infer.

Return JSON with these keys (null if not visible):
{
  "location": string,
  "date": string,
  "time": string,
  "device_model": string,
  "os_version": string,
  "build_version": string,
  "map_provider": string
}

Notes:
- Device model may be like "iPhone 11", "Pixel 7", etc.
- Build version may be numeric like "26.11"
- Only return values that are explicitly visible.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ]
            }]
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {}
    

def select_semantic_frame(frames):
    """
    Pick a frame likely to contain device/build info.
    Heuristic: later frames usually contain About/Version screens.
    """
    if not frames:
        return None
    return frames[-1]  # last frame is best for build/version


# ---------------------------
# MAIN
# ---------------------------
def main(api_key, video_paths):
    client = get_client(api_key)

    all_results = []  # store results for all videos

    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")

        # 1. Extract frames for THIS video
        frames = extract_frames(video_path)

        all_text = []

        # 2. OCR on all frames (fast + deterministic)
        for frame in frames:
            try:
                ocr_text = pytesseract.image_to_string(frame)
                all_text.extend(
                    [l.strip() for l in ocr_text.splitlines() if l.strip()]
                )
            except:
                pass

        # 3. Derive structured audit fields
        final = derive_fields(all_text)

        # 4. Add video name for traceability
        final["video_name"] = {
            "value": os.path.basename(video_path),
            "visible": True
        }

        # 5. Calculate risk
        risk, flags = calculate_risk(final)

        # 6. Attach risk to record
        final["_risk_level"] = risk
        final["_risk_flags"] = "; ".join(flags)

        # Debug print (VERY useful)
        print("FINAL DATA:")
        print(json.dumps(final, indent=2))

        all_results.append(final)

    # 7. Save combined outputs
    save_csv_multiple(all_results)

    print("\nALL VIDEOS PROCESSED SUCCESSFULLY")



    # ------------------------------------------------




# ---------------------------
# ENTRY POINTS
# ---------------------------
def run_agent(api_key, video_paths):
    main(api_key, video_paths)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    main(api_key)
