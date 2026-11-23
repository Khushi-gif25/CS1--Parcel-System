"""
Prozo | Parcel Intelligence System
Automated Parcel Image Processing & Shipment Data Extraction

Production-grade application providing:
- Bulk upload & validation of parcel photographs (JPG/PNG)
- Optical 1D & 2D Barcode Decoding (Code128, Code39, Code93, ITF, EAN, UPC, QR)
- OpenCV-based computer vision assessment & preprocessing
- EasyOCR text & confidence extraction
- Contextual Courier AWB, Weight, and Dimension extraction (with anti-false-positive filtering)
- AWB Cross-Validation (Barcode vs OCR: MATCHED, MISMATCH, BARCODE_ONLY, OCR_ONLY)
- Strict 9-tier status classification with honest manual review flags
- Interactive operations dashboard with search, filter, and visual inspector
- Downloadable CSV audit report with 1:1 image row mapping
"""

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

# ZXing C++ barcode decoding engine
try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

# Configure Streamlit page settings
st.set_page_config(
    page_title="Prozo | Parcel Intelligence System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. THEME & STYLING (Professional White/Light-Grey with Prozo Purple Accents)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global application styling */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header container */
    .prozo-header {
        background: linear-gradient(135deg, #2E1065 0%, #4C1D95 50%, #6D28D9 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.15);
    }
    .prozo-header h1 {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .prozo-header p {
        color: #DDD6FE !important;
        font-size: 15px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
        color: #1E293B;
    }
    .metric-card-total { border-left: 5px solid #6D28D9; }
    .metric-card-success { border-left: 5px solid #10B981; }
    .metric-card-review { border-left: 5px solid #F59E0B; }
    .metric-card-failed { border-left: 5px solid #EF4444; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-partial { background-color: #FEF3C7; color: #92400E; }
    .badge-warning { background-color: #FFEDD5; color: #9A3412; }
    .badge-danger { background-color: #FDE8E8; color: #9B1C1C; }
    .badge-info { background-color: #EDE9FE; color: #5B21B6; }
    .badge-match { background-color: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-mismatch { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-single { background-color: #E0E7FF; color: #3730A3; }

    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #6D28D9;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #5B21B6;
        color: white;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.25);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. OCR ENGINE INITIALIZATION (Cached Singleton)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing EasyOCR local engine...")
def get_ocr_reader():
    """
    Initializes EasyOCR reader once. Uses local model directory
    for offline execution without downloading or permission issues.
    """
    import easyocr
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    if os.path.isdir(model_dir):
        return easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=model_dir,
            user_network_directory=model_dir,
            download_enabled=False
        )
    return easyocr.Reader(['en'], gpu=False)


# -----------------------------------------------------------------------------
# 3. STEP 1: IMAGE VALIDATION
# -----------------------------------------------------------------------------
def validate_image(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[Image.Image], Optional[np.ndarray], str]:
    """
    Validates file integrity, dimensions, and readability.
    Returns (is_valid, pil_image, cv2_bgr_image, message).
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, None, None, "File is completely empty (0 bytes)."

    # Reject files larger than 35MB to safeguard system memory
    if len(file_bytes) > 35 * 1024 * 1024:
        return False, None, None, f"File size exceeds 35MB limit ({len(file_bytes) / (1024*1024):.1f}MB)."

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()  # Verify stream integrity
        # Reopen because verify() closes the stream
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, Exception) as exc:
        return False, None, None, f"Corrupted or invalid image file: {str(exc)}"

    w, h = pil_img.size
    if w < 60 or h < 60:
        return False, None, None, f"Image dimensions too small ({w}x{h} px); minimum required is 60x60 px."

    # Convert to OpenCV BGR format
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return True, pil_img, cv2_img, "Image is valid and readable."


# -----------------------------------------------------------------------------
# 4. STEP 2: PARCEL & LABEL ASSESSMENT (Heuristic Computer Vision)
# -----------------------------------------------------------------------------
def assess_parcel_and_label(cv2_img: np.ndarray) -> Dict[str, Any]:
    """
    Performs conservative geometric and contrast heuristics to assess parcel & label.
    """
    h, w = cv2_img.shape[:2]
    total_area = float(h * w)

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_dev = float(np.std(gray))

    # Extreme blur or blank image heuristic
    if std_dev < 10.0 or lap_var < 5.0:
        return {
            "parcel_detected": False,
            "multiple_parcels": False,
            "partial_parcel": False,
            "label_detected": False,
            "label_blocked": False,
            "note": "Low variance / uniform solid image; no textured object detected."
        }

    # Parcel contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Large contours representing potential parcel bodies (> 12% total area)
    large_contours = [c for c in contours if cv2.contourArea(c) > (0.12 * total_area)]
    multiple_parcels = len(large_contours) >= 2

    # Partial parcel check (contour truncated by camera boundary)
    partial_parcel = False
    if large_contours:
        primary_box = max(large_contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(primary_box)
        touch_left = bx <= 3
        touch_right = (bx + bw) >= (w - 3)
        touch_top = by <= 3
        touch_bottom = (by + bh) >= (h - 3)
        edge_touches = sum([touch_left, touch_right, touch_top, touch_bottom])
        if edge_touches >= 3 or ((touch_left or touch_right) and bw >= 0.85 * w):
            partial_parcel = True

    # Shipping label detection: shipping labels are high-brightness rectangular regions
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    label_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    label_detected = False
    label_blocked = False
    valid_labels = []

    for c in label_contours:
        area = cv2.contourArea(c)
        if (0.02 * total_area) <= area <= (0.80 * total_area):
            lx, ly, lw, lh = cv2.boundingRect(c)
            aspect_ratio = float(lw) / float(lh) if lh > 0 else 0
            if 0.3 <= aspect_ratio <= 3.5:
                valid_labels.append((lx, ly, lw, lh, area))

    if valid_labels:
        label_detected = True
        dominant_label = max(valid_labels, key=lambda x: x[4])
        lx, ly, lw, lh, _ = dominant_label
        label_roi = gray[ly:ly+lh, lx:lx+lw]
        if label_roi.size > 0:
            dark_pixels = np.sum(label_roi < 70)
            dark_ratio = float(dark_pixels) / float(label_roi.size)
            if dark_ratio > 0.35:
                label_blocked = True

    return {
        "parcel_detected": True,
        "multiple_parcels": multiple_parcels,
        "partial_parcel": partial_parcel,
        "label_detected": label_detected,
        "label_blocked": label_blocked,
        "note": "Geometric & contrast heuristic assessment."
    }


# -----------------------------------------------------------------------------
# 5. STEP 3: OPTICAL 1D & 2D BARCODE DECODING
# -----------------------------------------------------------------------------
def scan_barcodes(cv2_img: np.ndarray) -> Dict[str, Any]:
    """
    Robust multi-strategy barcode decoder supporting Code 128, Code 39, Code 93,
    ITF (Interleaved 2 of 5), EAN, UPC, and QR codes.
    Applies multi-angle rotations, grayscale, CLAHE, and scaling passes.
    Does NOT substitute OCR for barcode reading.
    """
    detected_raw = []
    seen_texts = set()

    # Define multi-pass image transformations for hard-to-read barcodes
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    passes = [
        ("original_bgr", cv2_img),
        ("grayscale", gray),
        ("clahe", clahe_img),
        ("rotated_90", cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)),
        ("rotated_270", cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    # Strategy 1: ZXing C++ Engine (Industry Standard)
    if HAS_ZXING:
        for pass_name, img_variant in passes:
            try:
                z_results = zxingcpp.read_barcodes(img_variant)
                for res in z_results:
                    txt = res.text.strip()
                    fmt = str(res.format).replace("BarcodeFormat.", "").replace("Format.", "")
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        detected_raw.append({
                            "format": fmt,
                            "data": txt,
                            "pass": pass_name
                        })
            except Exception:
                pass

    # Strategy 2: OpenCV Native Barcode & QR Detectors (Fallback)
    try:
        bd = cv2.barcode.BarcodeDetector()
        ok, d_info, d_type, _ = bd.detectAndDecodeWithType(cv2_img)
        if ok and d_info:
            for info, btype in zip(d_info, d_type):
                txt = str(info).strip()
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    detected_raw.append({
                        "format": str(btype) if btype else "1D_BARCODE",
                        "data": txt,
                        "pass": "opencv_barcode"
                    })
    except Exception:
        pass

    try:
        qd = cv2.QRCodeDetector()
        ok_q, q_info, _, _ = qd.detectAndDecodeMulti(cv2_img)
        if ok_q and q_info:
            for info in q_info:
                txt = str(info).strip()
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    detected_raw.append({
                        "format": "QR_CODE",
                        "data": txt,
                        "pass": "opencv_qr"
                    })
    except Exception:
        pass

    # Filter and rank courier AWB barcodes vs auxiliary internal order barcodes
    courier_awb_candidates = []
    for b in detected_raw:
        data = b["data"]
        clean_code = re.sub(r'[^A-Za-z0-9]', '', data)
        # Standard courier AWB barcodes are typically 8 to 20 alphanumeric characters
        # Short 4-6 digit barcodes are usually internal Order IDs or shelf bin tags
        length = len(clean_code)
        if 8 <= length <= 24:
            courier_awb_candidates.append({
                "format": b["format"],
                "data": clean_code.upper(),
                "priority": 100 if length >= 10 else 70
            })
        else:
            # Short barcodes (e.g. 6-digit Order ID 205611)
            courier_awb_candidates.append({
                "format": b["format"],
                "data": clean_code.upper(),
                "priority": 30
            })

    # Sort so true courier AWB (higher length/priority) comes first
    courier_awb_candidates.sort(key=lambda x: x["priority"], reverse=True)

    primary_awb = courier_awb_candidates[0]["data"] if courier_awb_candidates else None
    primary_format = courier_awb_candidates[0]["format"] if courier_awb_candidates else None

    return {
        "barcode_detected": len(detected_raw) > 0,
        "barcodes": detected_raw,
        "primary_awb": primary_awb,
        "primary_format": primary_format
    }


# -----------------------------------------------------------------------------
# 6. STEP 4: IMAGE PREPROCESSING USING OPENCV (FOR OCR)
# -----------------------------------------------------------------------------
def preprocess_image(cv2_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies image resizing (aspect-ratio preserved), grayscale conversion,
    noise filtering, CLAHE contrast enhancement, and sharpening for OCR.
    Returns (processed_for_ocr, resized_color_for_display).
    """
    h, w = cv2_img.shape[:2]
    max_dim = 1800.0
    min_dim = 600.0
    scale = 1.0

    current_max = max(h, w)
    if current_max > max_dim:
        scale = max_dim / current_max
    elif current_max < min_dim:
        scale = min_dim / current_max

    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        display_img = cv2.resize(cv2_img, (new_w, new_h), interpolation=interp)
    else:
        display_img = cv2_img.copy()

    # 1. Grayscale conversion
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)

    # 2. Noise reduction via bilateral filter (preserves sharp text boundaries)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 3. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Sharpening kernel
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    return sharpened, display_img


# -----------------------------------------------------------------------------
# 7. STEP 5: OCR USING EASYOCR
# -----------------------------------------------------------------------------
def extract_text(reader: Any, processed_img: np.ndarray) -> Tuple[str, List[str], float, List[Dict[str, Any]]]:
    """
    Performs OCR on preprocessed image using EasyOCR.
    Extracts text bounding boxes and confidence scores.
    Returns (raw_text, lines, avg_confidence, detailed_records).
    """
    try:
        ocr_results = reader.readtext(processed_img, paragraph=False)
    except Exception:
        return "", [], 0.0, []

    if not ocr_results:
        return "", [], 0.0, []

    cleaned_results = []
    lines = []
    confs = []

    for item in ocr_results:
        if len(item) == 3:
            bbox, text, conf = item
            text_str = str(text).strip()
            if text_str:
                conf_val = float(conf)
                lines.append(text_str)
                confs.append(conf_val)
                cleaned_results.append({
                    "bbox": bbox,
                    "text": text_str,
                    "confidence": round(conf_val, 4)
                })

    avg_conf = float(np.mean(confs)) if confs else 0.0
    raw_text = "\n".join(lines)
    return raw_text, lines, round(avg_conf, 4), cleaned_results


# -----------------------------------------------------------------------------
# 8. STEP 6: CONTEXTUAL COURIER AWB EXTRACTION
# -----------------------------------------------------------------------------
def extract_awb(text: str, lines: List[str]) -> Tuple[Optional[str], float, str]:
    """
    Contextual AWB extractor:
    - Prioritises numbers appearing near keywords such as Courier AWB No, AWB Number,
      Waybill Number, Tracking ID, Shipment ID.
    - Excludes phone/contact numbers, order IDs, pincodes, dates, prices.
    - Preserves source text and confidence.
    - If no reliable contextual candidate exists, returns (None, 0.0, "").
    """
    # Strong AWB Context Keywords
    awb_kw_regex = re.compile(
        r'(?i)\b(?:courier\s*awb(?:\s*no\.?|\s*na\.?|\s*number)?|awb(?:[_\s]*(?:no\.?|na\.?|number|num\.?|#))?|waybill(?:\s*no\.?|\s*na\.?|\s*number)?|tracking(?:\s*id|\s*no\.?|\s*number|\s*#)?|consignment(?:\s*no\.?)?|docket(?:\s*no\.?)?|shipment\s*id)\b'
    )
    # Unrelated keywords that must NEVER be extracted as AWB
    blacklist_kw_regex = re.compile(
        r'(?i)\b(?:contact(?:\s*number)?|phone|mobile|mob|tel|telephone|call|customer|cust|pin|pincode|zip|order\s*id|order#|item|po\s*no|po#|invoice|gst|gstin|tin|price|rs\.?|inr|qty|quantity|date|time|datetime|id\s*:)\b'
    )

    candidates = []

    for i, line in enumerate(lines):
        clean_line = line.strip()

        # Strategy A: Same line contains both AWB keyword and alphanumeric value
        # e.g., "AWB NO: PRZ9847201948" or "Courier AWB No: 77563772064"
        same_m = re.search(
            r'(?i)\b(?:courier\s*awb(?:\s*no\.?|\s*na\.?|\s*number)?|awb(?:[_\s]*(?:no\.?|na\.?|number|num\.?|#))?|waybill(?:\s*no\.?|\s*na\.?)?|tracking(?:\s*id|\s*no\.?)?|shipment\s*id)\s*[:#_\-]?\s*([A-Za-z0-9\-]{8,24})\b',
            clean_line
        )
        if same_m:
            candidate_val = same_m.group(1).strip("- ")
            digits_count = sum(c.isdigit() for c in candidate_val)
            if digits_count >= 6 and not blacklist_kw_regex.search(clean_line):
                candidates.append({
                    "val": candidate_val.upper(),
                    "score": 98,
                    "conf": 0.95,
                    "source": f"Same-line keyword: '{clean_line}'"
                })

        # Strategy B: AWB keyword is on line i, and the value is on line i + 1
        # e.g., Line i = "AWB_NO", Line i + 1 = "77563772064"
        if awb_kw_regex.search(clean_line) and not blacklist_kw_regex.search(clean_line):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Clean candidate
                candidate_val = re.sub(r'[^A-Za-z0-9\-]', '', next_line)
                digits_count = sum(c.isdigit() for c in candidate_val)
                # Ensure it's not a secondary keyword line
                if 8 <= len(candidate_val) <= 24 and digits_count >= 6 and not blacklist_kw_regex.search(next_line):
                    candidates.append({
                        "val": candidate_val.upper(),
                        "score": 92,
                        "conf": 0.90,
                        "source": f"Adjacent to keyword line: '{clean_line}' -> '{next_line}'"
                    })

    # Strategy C: Look for courier prefix codes (e.g. PRZ..., DEL..., EXP...) with digits
    if not candidates:
        for line in lines:
            if blacklist_kw_regex.search(line):
                continue
            matches = re.findall(r'\b(?:[A-Z]{2,4}[0-9]{8,14})\b', line)
            for m in matches:
                candidates.append({
                    "val": m.upper(),
                    "score": 75,
                    "conf": 0.75,
                    "source": f"Courier prefix pattern: '{line}'"
                })

    # Strict False-Positive Filtering:
    # Check if any candidate is actually a customer phone number or pincode
    valid_candidates = []
    for c in candidates:
        raw_digits = re.sub(r'\D', '', c["val"])
        # Check if 10-digit phone number
        if len(raw_digits) == 10 and raw_digits[0] in '6789':
            # Check if this exact number appears near 'Contact' or 'Phone' in OCR text
            is_phone = False
            for line in lines:
                if blacklist_kw_regex.search(line) and raw_digits in re.sub(r'\D', '', line):
                    is_phone = True
                    break
            if is_phone:
                continue  # Disqualify contact phone numbers

        # Disqualify 6-digit Indian PIN codes
        if len(raw_digits) == 6:
            continue

        valid_candidates.append(c)

    if valid_candidates:
        valid_candidates.sort(key=lambda x: (x["score"], x["conf"]), reverse=True)
        top = valid_candidates[0]
        return top["val"], top["conf"], top["source"]

    # If no reliable candidate with context exists, return empty and flag for manual review
    return None, 0.0, ""


# -----------------------------------------------------------------------------
# 9. STEP 7: STRUCTURED WEIGHT & DIMENSIONS EXTRACTION
# -----------------------------------------------------------------------------
def extract_weight(text: str, lines: List[str]) -> Tuple[Optional[str], float, str]:
    """
    Extracts parcel weight in KG or Grams.
    Supports both single-line ('WEIGHT: 1.45 KG') and multi-line ('WEIGHT(Kg)' -> '1.26').
    Returns (weight_str, confidence, source).
    """
    # Pass 1: Multi-line e.g. "WEIGHT(Kg)" on line i, "1.26" on line i + 1
    kw_re = re.compile(r'(?i)\b(?:GROSS\s*WT|ACTUAL\s*WT|DEAD\s*WT|NET\s*WT|WEIGHT|WT)\s*(?:\((?:KG|KGS|G|GM)\))?\s*[:\-]?\s*$')
    for i, line in enumerate(lines[:-1]):
        if kw_re.search(line.strip()):
            nxt = lines[i + 1].strip()
            m_num = re.search(r'^([0-9]+(?:\.[0-9]+)?)\s*(KG|KGS|G|GM)?$', nxt, re.IGNORECASE)
            if m_num:
                val = float(m_num.group(1))
                unit = m_num.group(2).upper() if m_num.group(2) else ("G" if "g" in line.lower() and "kg" not in line.lower() else "KG")
                if unit in ["G", "GM", "GRAMS"]:
                    if val >= 1000.0:
                        kg_val = val / 1000.0
                        return f"{kg_val:.2f} KG", 0.95, f"{line} -> {nxt}"
                    elif 10.0 <= val < 1000.0:
                        return f"{int(val) if val.is_integer() else val} G", 0.95, f"{line} -> {nxt}"
                elif unit in ["KG", "KGS"] and 0.01 <= val <= 250.0:
                    return f"{val:.2f} KG", 0.95, f"{line} -> {nxt}"

    # Pass 2: Single-line keyword-associated weight
    kw_pattern = re.compile(
        r'(?i)\b(?:GROSS\s*WT|ACTUAL\s*WT|DEAD\s*WT|NET\s*WT|CHARGEABLE\s*WT|CHG\s*WT|WEIGHT|WT)\s*(?:\((?:KG|KGS|G|GM)\))?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(KG|KGS|G|GM|GRAMS|LBS)?\b'
    )
    for line in lines:
        m = kw_pattern.search(line)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper() if m.group(2) else "KG"
            if unit in ["G", "GM", "GRAMS"]:
                if val >= 1000.0:
                    kg_val = val / 1000.0
                    return f"{kg_val:.2f} KG", 0.95, line
                elif 10.0 <= val < 1000.0:
                    return f"{int(val) if val.is_integer() else val} G", 0.95, line
            elif unit in ["KG", "KGS"] and 0.01 <= val <= 250.0:
                return f"{val:.2f} KG", 0.95, line

    # Pass 3: Standalone numbers with weight units (avoiding dimension strings)
    standalone_regex = re.compile(r'(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(KG|KGS|GM|GMS|GRAMS)\b')
    for line in lines:
        if re.search(r'(?i)[0-9]+\s*[xX*×]\s*[0-9]+', line):
            continue
        m = standalone_regex.search(line)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit in ["GM", "GMS", "GRAMS"]:
                if val >= 1000.0:
                    kg_val = val / 1000.0
                    return f"{kg_val:.2f} KG", 0.85, line
                elif 10.0 <= val < 1000.0:
                    return f"{int(val) if val.is_integer() else val} G", 0.85, line
            elif unit in ["KG", "KGS"] and 0.01 <= val <= 250.0:
                return f"{val:.2f} KG", 0.85, line

    return None, 0.0, ""


def extract_dimensions(text: str, lines: List[str]) -> Tuple[Optional[str], float, str]:
    """
    Extracts parcel dimensions (L x W x H) in CM.
    Supports both multi-line ('LENGTH(cm)' -> '35.00', 'WIDTH' -> '24.90', 'HEIGHT' -> '12.20')
    and single-line formats ('30 x 20 x 15 CM').
    Returns (dimensions_str, confidence, source).
    """
    # Pass 1: Multi-line / Labelled LENGTH / WIDTH / HEIGHT:
    l_val, w_val, h_val = None, None, None
    for i, line in enumerate(lines):
        clean = line.strip().upper()
        # Same-line check: e.g. "Length: 57.000 cm"
        m_l = re.search(r'\bLENGTH(?:\(CM\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)', clean)
        if m_l and l_val is None: l_val = float(m_l.group(1))
        m_w = re.search(r'\bWIDTH(?:\(CM\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)', clean)
        if m_w and w_val is None: w_val = float(m_w.group(1))
        m_h = re.search(r'\bHEIGHT(?:\(CM\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)', clean)
        if m_h and h_val is None: h_val = float(m_h.group(1))

        # Adjacent-line check if not matched on same line: e.g. "Length:" -> "23.000 cm"
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if l_val is None and re.search(r'^\s*LENGTH(?:\(CM\))?[:\s]*$', clean):
                m = re.search(r'^([0-9]+(?:\.[0-9]+)?)\b', nxt)
                if m: l_val = float(m.group(1))
            elif w_val is None and re.search(r'^\s*WIDTH(?:\(CM\))?[:\s]*$', clean):
                m = re.search(r'^([0-9]+(?:\.[0-9]+)?)\b', nxt)
                if m: w_val = float(m.group(1))
            elif h_val is None and re.search(r'^\s*HEIGHT(?:\(CM\))?[:\s]*$', clean):
                m = re.search(r'^([0-9]+(?:\.[0-9]+)?)\b', nxt)
                if m: h_val = float(m.group(1))

    if l_val is not None and w_val is not None and h_val is not None:
        if 1.0 <= l_val <= 500.0 and 1.0 <= w_val <= 500.0 and 1.0 <= h_val <= 500.0:
            l_s = f"{int(l_val) if l_val.is_integer() else l_val}"
            w_s = f"{int(w_val) if w_val.is_integer() else w_val}"
            h_s = f"{int(h_val) if h_val.is_integer() else h_val}"
            return f"{l_s} x {w_s} x {h_s} CM", 0.95, "Labelled LENGTH/WIDTH/HEIGHT"

    # Pass 2: Single-line 3D pattern (30 x 20 x 15 CM)
    dim_3d_regex = re.compile(
        r'(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(?:[xX*×]|by)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[xX*×]|by)\s*([0-9]+(?:\.[0-9]+)?)\s*(CM|MM|IN|INCHES)?\b'
    )
    for line in lines:
        m = dim_3d_regex.search(line)
        if m:
            l, w, h = float(m.group(1)), float(m.group(2)), float(m.group(3))
            unit = m.group(4).upper() if m.group(4) else "CM"
            if 1.0 <= l <= 500.0 and 1.0 <= w <= 500.0 and 1.0 <= h <= 500.0:
                l_s = f"{int(l) if l.is_integer() else l}"
                w_s = f"{int(w) if w.is_integer() else w}"
                h_s = f"{int(h) if h.is_integer() else h}"
                return f"{l_s} x {w_s} x {h_s} {unit}", 0.95, line

    # Pass 3: L: ... W: ... H: ... pattern
    dim_lwh_regex = re.compile(
        r'(?i)\bL\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*[,\s]*W\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*[,\s]*H\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(CM|MM)?\b'
    )
    for line in lines:
        m = dim_lwh_regex.search(line)
        if m:
            l, w, h = float(m.group(1)), float(m.group(2)), float(m.group(3))
            unit = m.group(4).upper() if m.group(4) else "CM"
            if 1.0 <= l <= 500.0 and 1.0 <= w <= 500.0 and 1.0 <= h <= 500.0:
                l_s = f"{int(l) if l.is_integer() else l}"
                w_s = f"{int(w) if w.is_integer() else w}"
                h_s = f"{int(h) if h.is_integer() else h}"
                return f"{l_s} x {w_s} x {h_s} {unit}", 0.90, line

    return None, 0.0, ""


# -----------------------------------------------------------------------------
# 10. STEP 8: LOCATION EXTRACTION
# -----------------------------------------------------------------------------
def extract_location(text: str, lines: List[str]) -> Tuple[Optional[str], float, str]:
    """
    Extracts location information printed on the parcel label.
    Looks for contextual keywords such as Location:, Hub:, Warehouse:, Facility:,
    Dispatch Location:, Origin:.
    Preserves complete location name, underscores, parentheses, and state info.
    Distinguishes from customer/delivery addresses.
    Returns (location_str, confidence, source).
    """
    blacklist = re.compile(
        r'(?i)\b(?:delivery\s*address|customer|consignee|destination|receiver|ship\s*to|bill\s*to|return\s*address)\b'
    )
    kw_re = re.compile(
        r'(?i)^\s*(?:dispatch\s*location|origin(?:\s*location|\s*hub)?|location|hub|warehouse|facility)\s*[:#_\-]?\s*$'
    )

    # Strategy 1: Same line keyword + location value
    # e.g., "Location: Bangalore_Hoskote_GW (Karnataka)"
    for line in lines:
        if blacklist.search(line):
            continue
        m = re.search(
            r'(?i)\b(?:dispatch\s*location|origin|location|hub|warehouse|facility)\s*[:#_\-]\s*([A-Za-z0-9_\-\s(),.\[\]{}j!]+)',
            line
        )
        if m:
            val = m.group(1).strip()
            # Clean trailing parenthesis OCR distortion e.g. (Karnataka[ or (Karnatakaj -> (Karnataka)
            val = re.sub(r'\(([A-Za-z0-9_\-\s]+)[j\]\}1|!]\s*$', r'(\1)', val)
            if len(val) >= 3 and not re.match(r'(?i)^(?:date|time|awb|weight|length|width|height)', val):
                return val, 0.95, line

    # Strategy 2: Keyword on line i, location value on line i + 1
    # e.g., Line i = "Location:", Line i + 1 = "INMHAPVP" or "Bangalore_Hoskote_GW (Karnataka)"
    for i, line in enumerate(lines[:-1]):
        if blacklist.search(line):
            continue
        if kw_re.search(line.strip()):
            nxt = lines[i + 1].strip()
            nxt = re.sub(r'\(([A-Za-z0-9_\-\s]+)[j\]\}1|!]\s*$', r'(\1)', nxt)
            if not re.match(r'(?i)^(?:awb|length|width|height|weight|time|date|rvol|order|contact|phone|location|hub):?', nxt):
                if len(nxt) >= 3:
                    return nxt, 0.90, f"{line} -> {nxt}"

    return None, 0.0, ""


# -----------------------------------------------------------------------------
# 11. STEP 9: DATE & TIME EXTRACTION
# -----------------------------------------------------------------------------
def extract_datetime(text: str, lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], float, str]:
    """
    Extracts Date and Time values printed on parcel labels.
    Supports 12-hour (AM/PM) and 24-hour formats with seconds.
    Normalizes date to YYYY-MM-DD.
    Preserves original raw timestamp.
    Returns (normalized_date, normalized_time, raw_timestamp, confidence, source).
    """
    dt_kw = re.compile(r'(?i)\b(?:shipment\s*date|dispatch\s*date|booking\s*date|date\s*time|datetime|date|time)\b')
    blacklist = re.compile(r'(?i)\b(?:delivery\s*date|order\s*date|expiry|exp\s*date)\b')

    raw_candidates = []

    # Strategy 1: Same line keyword + timestamp
    # e.g. "Time: 7/13/2025 5:53:26 PM"
    for line in lines:
        if blacklist.search(line):
            continue
        m = re.search(r'(?i)\b(?:shipment\s*date|dispatch\s*date|booking\s*date|datetime|date|time)\s*[:#_\-]\s*(.+)', line)
        if m:
            val = m.group(1).strip()
            if re.search(r'\d{1,4}[^\d\s]\d{1,2}[^\d\s]\d{2,4}|\d{1,2}[:.]\d{2}', val):
                raw_candidates.append((val, 0.95, line))

    # Strategy 2: Keyword on line i, timestamp on line i + 1
    # e.g. Line i = "Time:", Line i + 1 = "7/20/2025 12.57.40 AM"
    for i, line in enumerate(lines[:-1]):
        if blacklist.search(line):
            continue
        clean = line.strip()
        if dt_kw.search(clean) and len(clean) <= 20:
            nxt = lines[i + 1].strip()
            if re.search(r'\d{1,4}[^\d\s]\d{1,2}[^\d\s]\d{2,4}|\d{1,2}[:.]\d{2}', nxt):
                raw_candidates.append((nxt, 0.90, f"{clean} -> {nxt}"))

    # Strategy 3: Scan for standalone timestamps if no keyword was explicit
    if not raw_candidates:
        for line in lines:
            if re.search(r'\b\d{1,2}[^\d\s]\d{1,2}[^\d\s]\d{4}\s+\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(?:AM|PM)?\b', line, re.IGNORECASE):
                raw_candidates.append((line.strip(), 0.70, line.strip()))

    if not raw_candidates:
        return None, None, None, 0.0, ""

    # Parse best candidate
    for raw_str, conf, source in raw_candidates:
        # Pre-clean common OCR artifacts:
        # 1. Glued year and time: "20255.53.26" -> "2025 5.53.26"
        clean_raw = re.sub(r'(\d{4})(\d{1,2}[:.]\d{2})', r'\1 \2', raw_str)
        # 2. Fix OCR separators between date parts: "7{13/2025" or "7'13/2025" -> "7/13/2025"
        clean_raw = re.sub(r'(\d{1,2})[^\d\s](\d{1,2})[^\d\s](\d{4})', r'\1/\2/\3', clean_raw)
        clean_raw = re.sub(r'(\d{4})[^\d\s](\d{1,2})[^\d\s](\d{1,2})', r'\1-\2-\3', clean_raw)

        norm_date = None
        norm_time = None

        # Date normalization to YYYY-MM-DD
        date_m = re.search(r'\b(\d{1,4})[-/\'.](\d{1,2})[-/\'.](\d{2,4})\b', clean_raw)
        if date_m:
            p1, p2, p3 = int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3))
            try:
                if p1 > 1000:  # YYYY-MM-DD
                    year, month, day = p1, p2, p3
                elif p3 > 1000:
                    year = p3
                    if p1 > 12:  # DD/MM/YYYY
                        day, month = p1, p2
                    elif p2 > 12:  # MM/DD/YYYY
                        month, day = p1, p2
                    else:  # Common logistics US format M/D/YYYY
                        month, day = p1, p2
                norm_date = f"{year:04d}-{month:02d}-{day:02d}"
            except Exception:
                pass

        # Time normalization to HH:MM:SS AM/PM or HH:MM:SS
        time_m = re.search(r'\b(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?(?:\.\d+)?\s*(AM|PM)?\b', clean_raw, re.IGNORECASE)
        if time_m:
            h = int(time_m.group(1))
            m = int(time_m.group(2))
            s = int(time_m.group(3)) if time_m.group(3) else 0
            ampm = time_m.group(4).upper() if time_m.group(4) else None
            if ampm:
                norm_time = f"{h:02d}:{m:02d}:{s:02d} {ampm}"
            else:
                norm_time = f"{h:02d}:{m:02d}:{s:02d}"

        if norm_date or norm_time:
            return norm_date, norm_time, raw_str, conf, source

    return None, None, None, 0.0, ""


# -----------------------------------------------------------------------------
# 10. STEP 8: AWB CROSS-VALIDATION (BARCODE vs OCR)
# -----------------------------------------------------------------------------
def cross_validate_awb(
    awb_barcode: Optional[str],
    barcode_format: Optional[str],
    awb_ocr: Optional[str],
    ocr_conf: float
) -> Dict[str, Any]:
    """
    Compares the AWB obtained through OCR with the barcode-decoded value:
    - If both match -> MATCHED.
    - If they differ -> MISMATCH (requires manual verification).
    - If only one succeeds -> BARCODE_ONLY or OCR_ONLY (cross-validation unavailable).
    - Never silently overwrites one source with another.
    """
    has_barcode = bool(awb_barcode and awb_barcode.strip())
    has_ocr = bool(awb_ocr and awb_ocr.strip())

    if has_barcode and has_ocr:
        # Normalize comparison (remove punctuation, uppercase)
        clean_b = re.sub(r'[^A-Za-z0-9]', '', awb_barcode).upper()
        clean_o = re.sub(r'[^A-Za-z0-9]', '', awb_ocr).upper()

        if clean_b == clean_o or clean_b in clean_o or clean_o in clean_b:
            return {
                "verification": "MATCHED",
                "final_awb": awb_barcode,
                "confidence": 1.0,
                "note": f"Verified match between {barcode_format or 'Barcode'} and OCR."
            }
        else:
            return {
                "verification": "MISMATCH",
                "final_awb": f"{awb_barcode} (Barcode) / {awb_ocr} (OCR)",
                "confidence": 0.30,
                "note": f"Conflicting values: Barcode={awb_barcode} vs OCR={awb_ocr}. Manual verification required."
            }

    elif has_barcode:
        return {
            "verification": "BARCODE_ONLY",
            "final_awb": awb_barcode,
            "confidence": 0.95,
            "note": f"Decoded from {barcode_format or 'Barcode'} (OCR cross-validation unavailable)."
        }

    elif has_ocr:
        return {
            "verification": "OCR_ONLY",
            "final_awb": awb_ocr,
            "confidence": ocr_conf,
            "note": "Extracted via OCR (Barcode cross-validation unavailable)."
        }

    return {
        "verification": "NOT_FOUND",
        "final_awb": None,
        "confidence": 0.0,
        "note": "Courier AWB could not be identified from Barcode or OCR."
    }


# -----------------------------------------------------------------------------
# 11. STEP 9: EXTRACTION VALIDATION & 9-TIER STATUS CLASSIFICATION
# -----------------------------------------------------------------------------
def classify_status(
    is_valid: bool,
    validation_msg: str,
    cv_assessment: Dict[str, Any],
    raw_text: str,
    avg_confidence: float,
    awb_validation: Dict[str, Any],
    weight: Optional[str],
    dimensions: Optional[str]
) -> Tuple[str, str]:
    """
    Applies strict status precedence hierarchy across 9 categories.
    Guarantees LOW_CONFIDENCE on uncertain extractions, mismatches, or missing AWB.
    Never marks SUCCESS simply because a barcode was detected.
    """
    # Precedence 1: Image file integrity
    if not is_valid:
        return "INVALID_IMAGE", validation_msg

    # Precedence 2: Parcel presence (conservative heuristic)
    if not cv_assessment.get("parcel_detected", True):
        return "NO_PARCEL", "No parcel contour or textured object detected in scanning area."

    # Precedence 3: Multiple parcels
    if cv_assessment.get("multiple_parcels", False):
        return "MULTIPLE_PARCELS", "Multiple parcels detected in scanning area. Single parcel per scan required."

    # Precedence 4: Truncated / partial parcel
    if cv_assessment.get("partial_parcel", False):
        return "PARTIAL_PARCEL", "Parcel appears truncated at image frame edges; reposition parcel."

    # Precedence 5: Obstructed or covered label
    if cv_assessment.get("label_blocked", False):
        return "LABEL_BLOCKED", "Shipping label appears partially obstructed, occluded, or shadowed."

    # Precedence 6: Unreadable label (no OCR text produced and no barcode)
    if (not raw_text.strip() or len(raw_text.strip()) < 5) and awb_validation["verification"] == "NOT_FOUND":
        return "LABEL_UNREADABLE", "Parcel detected, but shipping label text and barcode could not be read."

    # Precedence 7: Mismatch between Barcode and OCR -> LOW_CONFIDENCE (Manual Review Required)
    if awb_validation["verification"] == "MISMATCH":
        return "LOW_CONFIDENCE", awb_validation["note"]

    # Precedence 8: Missing or unidentifiable AWB -> LOW_CONFIDENCE (Manual Review Required)
    if awb_validation["verification"] == "NOT_FOUND" or not awb_validation["final_awb"]:
        return "LOW_CONFIDENCE", "Courier AWB number could not be confidently identified; manual review required."

    # Precedence 9: Low overall OCR recognition confidence (< 40%)
    if avg_confidence < 0.40 and awb_validation["verification"] == "OCR_ONLY":
        return "LOW_CONFIDENCE", f"OCR recognition confidence is low ({avg_confidence:.1%}); manual verification required."

    final_awb = awb_validation["final_awb"]
    has_awb = bool(final_awb and final_awb.strip())
    has_wt = bool(weight and weight.strip())
    has_dims = bool(dimensions and dimensions.strip())

    # Precedence 10: Complete SUCCESS (AWB, Weight, Dimensions present & verified)
    if has_awb and has_wt and has_dims:
        return "SUCCESS", f"Shipment information successfully extracted & verified ({awb_validation['verification']})."

    # Precedence 11: Partial extraction (retain all found fields honestly)
    if has_awb or has_wt or has_dims:
        missing = []
        if not has_awb: missing.append("AWB")
        if not has_wt: missing.append("Weight")
        if not has_dims: missing.append("Dimensions")
        return "PARTIAL_EXTRACTION", f"Partial extraction: {', '.join(missing)} missing or unreadable on label."

    # Fallback: Text extracted, but no shipment fields could be identified
    return "LOW_CONFIDENCE", "Text extracted, but courier shipment fields could not be confidently identified."


# -----------------------------------------------------------------------------
# 12. SINGLE IMAGE PROCESSING
# -----------------------------------------------------------------------------
def process_single_image(reader: Any, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Coordinates end-to-end processing for an individual image.
    Guarantees exactly one record output and isolates exceptions.
    """
    record: Dict[str, Any] = {
        "image_filename": filename,
        "awb_number": None,
        "awb_verification": "NOT_FOUND",
        "barcode_format": None,
        "awb_barcode": None,
        "awb_ocr": None,
        "weight": None,
        "dimensions": None,
        "location": None,
        "shipment_date": None,
        "shipment_time": None,
        "raw_timestamp": None,
        "ocr_confidence": 0.0,
        "status": "INVALID_IMAGE",
        "reason": "",
        "extracted_text": "",
        "original_image": None,
        "preprocessed_image": None,
        "barcode_info": None,
        "awb_ocr_source": ""
    }

    try:
        # Step 1: Validate image
        is_valid, pil_img, cv2_img, val_msg = validate_image(file_bytes, filename)
        if not is_valid or cv2_img is None or pil_img is None:
            record["status"] = "INVALID_IMAGE"
            record["reason"] = val_msg
            return record

        record["original_image"] = pil_img

        # Step 2: Parcel and label assessment
        cv_assessment = assess_parcel_and_label(cv2_img)

        # Step 3: Optical Barcode & QR Code Scanning (1D & 2D)
        barcode_res = scan_barcodes(cv2_img)
        record["barcode_info"] = barcode_res
        record["awb_barcode"] = barcode_res.get("primary_awb")
        record["barcode_format"] = barcode_res.get("primary_format")

        # Step 4: OpenCV Preprocessing for OCR
        processed_img, display_img = preprocess_image(cv2_img)
        record["preprocessed_image"] = Image.fromarray(processed_img)

        # Step 5: OCR extraction
        raw_text, lines, avg_conf, _ = extract_text(reader, processed_img)
        record["extracted_text"] = raw_text
        record["ocr_confidence"] = avg_conf

        # Step 6: Contextual Extraction from OCR
        awb_ocr, ocr_awb_conf, ocr_awb_src = extract_awb(raw_text, lines)
        wt_ocr, _, _ = extract_weight(raw_text, lines)
        dims_ocr, _, _ = extract_dimensions(raw_text, lines)
        loc, _, _ = extract_location(raw_text, lines)
        s_date, s_time, raw_ts, _, _ = extract_datetime(raw_text, lines)

        record["awb_ocr"] = awb_ocr
        record["awb_ocr_source"] = ocr_awb_src
        record["weight"] = wt_ocr
        record["dimensions"] = dims_ocr
        record["location"] = loc
        record["shipment_date"] = s_date
        record["shipment_time"] = s_time
        record["raw_timestamp"] = raw_ts

        # Step 7: AWB Cross-Validation (Barcode vs OCR)
        awb_val = cross_validate_awb(
            awb_barcode=record["awb_barcode"],
            barcode_format=record["barcode_format"],
            awb_ocr=awb_ocr,
            ocr_conf=ocr_awb_conf
        )
        record["awb_number"] = awb_val["final_awb"]
        record["awb_verification"] = awb_val["verification"]

        # Step 8: Status Classification
        status, reason = classify_status(
            is_valid=True,
            validation_msg="",
            cv_assessment=cv_assessment,
            raw_text=raw_text,
            avg_confidence=avg_conf,
            awb_validation=awb_val,
            weight=wt_ocr,
            dimensions=dims_ocr
        )
        record["status"] = status
        record["reason"] = reason

    except Exception as exc:
        record["status"] = "INVALID_IMAGE"
        record["reason"] = f"Unexpected processing error: {str(exc)}"

    return record


# -----------------------------------------------------------------------------
# 13. BATCH PROCESSING PIPELINE
# -----------------------------------------------------------------------------
def process_batch(reader: Any, uploaded_files: List[Any], progress_bar: Any, status_text: Any) -> List[Dict[str, Any]]:
    """
    Sequentially processes a bulk list of images with live progress updates.
    Individual failures do not halt batch execution.
    """
    total = len(uploaded_files)
    results = []

    for idx, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        status_text.markdown(f"**Processing {idx + 1} of {total}:** `{filename}`")
        progress_bar.progress((idx + 1) / total)

        file_bytes = uploaded_file.getvalue()
        rec = process_single_image(reader, file_bytes, filename)
        results.append(rec)

    status_text.markdown(f"✅ **Batch processing completed:** {total} images processed.")
    return results


# -----------------------------------------------------------------------------
# 14. CSV GENERATION
# -----------------------------------------------------------------------------
def generate_csv(results: List[Dict[str, Any]]) -> str:
    """
    Generates standardized CSV with exactly 1 row per uploaded image.
    Uses RFC-4180 compliant escaping via pandas.
    """
    rows = []
    for r in results:
        rows.append({
            "image_filename": r.get("image_filename", ""),
            "awb_number": r.get("awb_number") or "",
            "awb_verification": r.get("awb_verification", ""),
            "barcode_format": r.get("barcode_format") or "",
            "awb_barcode": r.get("awb_barcode") or "",
            "awb_ocr": r.get("awb_ocr") or "",
            "weight": r.get("weight") or "",
            "dimensions": r.get("dimensions") or "",
            "location": r.get("location") or "",
            "shipment_date": r.get("shipment_date") or "",
            "shipment_time": r.get("shipment_time") or "",
            "raw_timestamp": r.get("raw_timestamp") or "",
            "ocr_confidence": f"{r.get('ocr_confidence', 0.0):.2f}",
            "status": r.get("status", ""),
            "reason": r.get("reason", ""),
            "extracted_text": r.get("extracted_text", "").replace("\n", " | ")
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# -----------------------------------------------------------------------------
# 15. STREAMLIT APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main():
    # Application Header
    st.markdown("""
    <div class="prozo-header">
        <h1>Prozo | Parcel Intelligence System</h1>
        <p>Automated Parcel Image Processing & Shipment Data Extraction</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for batch results
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # Load OCR Reader
    reader = get_ocr_reader()

    # --- Upload Section ---
    st.subheader("1. Bulk Parcel Photograph Upload")
    upload_col1, upload_col2 = st.columns([3, 1])

    with upload_col1:
        uploaded_files = st.file_uploader(
            "Select parcel photographs (JPG, JPEG, PNG). Supports hundreds of images in bulk:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload parcel scanning station photos containing courier labels."
        )

    with upload_col2:
        if uploaded_files:
            st.metric("Images Uploaded", len(uploaded_files))
            process_btn = st.button("🚀 Process Images", use_container_width=True)
            clear_btn = st.button("🔄 Reset Results", use_container_width=True)
            if clear_btn:
                st.session_state["results"] = None
                st.rerun()
        else:
            st.info("Awaiting image selection.")
            process_btn = False

    # Show list of uploaded filenames if available
    if uploaded_files:
        with st.expander(f"📁 View Uploaded Files ({len(uploaded_files)} files)", expanded=False):
            file_names_df = pd.DataFrame([
                {"#": i + 1, "Filename": f.name, "Size (KB)": round(f.size / 1024, 1)}
                for i, f in enumerate(uploaded_files)
            ])
            st.dataframe(file_names_df, use_container_width=True, height=180)

    # Trigger Processing
    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one JPG or PNG parcel image.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            with st.spinner("Processing image batch..."):
                results = process_batch(reader, uploaded_files, progress_bar, status_text)
                st.session_state["results"] = results
            st.success("Batch processing complete!")
            st.rerun()

    # --- Results Dashboard ---
    results = st.session_state.get("results")
    if results:
        st.markdown("---")
        st.subheader("2. Operations Dashboard & Analysis")

        # Compute metric aggregates
        total_count = len(results)
        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        review_count = sum(1 for r in results if r["status"] in ["PARTIAL_EXTRACTION", "LOW_CONFIDENCE", "LABEL_BLOCKED"])
        failed_count = sum(1 for r in results if r["status"] in ["NO_PARCEL", "MULTIPLE_PARCELS", "PARTIAL_PARCEL", "LABEL_UNREADABLE", "INVALID_IMAGE"])

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card metric-card-total">
                <div class="metric-label">Total Images</div>
                <div class="metric-value">{total_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card metric-card-success">
                <div class="metric-label">Successfully Processed</div>
                <div class="metric-value" style="color:#059669;">{success_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card metric-card-review">
                <div class="metric-label">Requiring Review</div>
                <div class="metric-value" style="color:#D97706;">{review_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card metric-card-failed">
                <div class="metric-label">Failed / Irregular</div>
                <div class="metric-value" style="color:#DC2626;">{failed_count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # CSV Download Section
        csv_data = generate_csv(results)
        csv_col1, csv_col2 = st.columns([3, 1])
        with csv_col1:
            st.caption(f"CSV report ready with exactly {len(results)} rows matching all uploaded images.")
        with csv_col2:
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name="prozo_shipment_data_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Filters and Search
        st.write("")
        f1, f2 = st.columns([1, 2])
        with f1:
            all_statuses = ["ALL", "SUCCESS", "PARTIAL_EXTRACTION", "LOW_CONFIDENCE", "LABEL_BLOCKED", "LABEL_UNREADABLE", "MULTIPLE_PARCELS", "PARTIAL_PARCEL", "NO_PARCEL", "INVALID_IMAGE"]
            selected_status = st.selectbox("Filter by Status:", all_statuses, index=0)
        with f2:
            search_query = st.text_input("Search by AWB Number or Filename:", placeholder="e.g. 77563772064, parcel_001.jpg").strip().lower()

        # Filter rows
        filtered_results = []
        for r in results:
            match_status = (selected_status == "ALL") or (r["status"] == selected_status)
            match_search = True
            if search_query:
                filename_match = search_query in r["image_filename"].lower()
                awb_match = r["awb_number"] is not None and search_query in str(r["awb_number"]).lower()
                match_search = filename_match or awb_match
            if match_status and match_search:
                filtered_results.append(r)

        # Render Results Table
        table_data = []
        for r in filtered_results:
            table_data.append({
                "Image Filename": r["image_filename"],
                "AWB Number": r["awb_number"] if r["awb_number"] else "N/A",
                "AWB Verification": r.get("awb_verification", "NOT_FOUND"),
                "Location": r.get("location") if r.get("location") else "N/A",
                "Date": r.get("shipment_date") if r.get("shipment_date") else "N/A",
                "Time": r.get("shipment_time") if r.get("shipment_time") else "N/A",
                "Weight": r["weight"] if r["weight"] else "N/A",
                "Dimensions": r["dimensions"] if r["dimensions"] else "N/A",
                "OCR Confidence": f"{r['ocr_confidence']:.2f}",
                "Status": r["status"],
                "Reason": r["reason"]
            })

        st.markdown(f"**Showing {len(filtered_results)} of {len(results)} records:**")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=280)

        # --- Single Record Inspection ---
        st.markdown("---")
        st.subheader("3. Single Image Record Inspector")

        if filtered_results:
            selected_filename = st.selectbox(
                "Select image record to inspect:",
                options=[r["image_filename"] for r in filtered_results]
            )
            # Find selected record
            selected_record = next(r for r in results if r["image_filename"] == selected_filename)

            insp_c1, insp_c2, insp_c3 = st.columns([1, 1, 1])

            with insp_c1:
                st.markdown("**Original Photograph**")
                if selected_record["original_image"]:
                    st.image(selected_record["original_image"], use_container_width=True)
                else:
                    st.warning("Original image unavailable (invalid or corrupt file).")

            with insp_c2:
                st.markdown("**OpenCV Preprocessed Image (OCR Feed)**")
                if selected_record["preprocessed_image"]:
                    st.image(selected_record["preprocessed_image"], use_container_width=True)
                else:
                    st.info("Preprocessed view not generated.")

            with insp_c3:
                st.markdown("**Extracted Shipment Metadata**")
                status = selected_record["status"]
                badge_class = "badge-info"
                if status == "SUCCESS": badge_class = "badge-success"
                elif status == "PARTIAL_EXTRACTION": badge_class = "badge-partial"
                elif status in ["LOW_CONFIDENCE", "LABEL_BLOCKED"]: badge_class = "badge-warning"
                else: badge_class = "badge-danger"

                verif = selected_record.get("awb_verification", "NOT_FOUND")
                verif_class = "badge-single"
                if verif == "MATCHED": verif_class = "badge-match"
                elif verif == "MISMATCH": verif_class = "badge-mismatch"

                awb_disp = selected_record['awb_number'] if selected_record['awb_number'] else '<span style="color:#94A3B8">N/A</span>'
                loc_disp = selected_record.get('location') if selected_record.get('location') else '<span style="color:#94A3B8">N/A</span>'
                date_disp = selected_record.get('shipment_date') if selected_record.get('shipment_date') else '<span style="color:#94A3B8">N/A</span>'
                time_disp = selected_record.get('shipment_time') if selected_record.get('shipment_time') else '<span style="color:#94A3B8">N/A</span>'
                raw_ts_disp = selected_record.get('raw_timestamp') if selected_record.get('raw_timestamp') else '<span style="color:#94A3B8">N/A</span>'
                wt_disp = selected_record['weight'] if selected_record['weight'] else '<span style="color:#94A3B8">N/A</span>'
                dims_disp = selected_record['dimensions'] if selected_record['dimensions'] else '<span style="color:#94A3B8">N/A</span>'

                b_val = selected_record.get('awb_barcode')
                b_fmt = selected_record.get('barcode_format')
                if b_val:
                    barcode_disp = f'<span style="color:#059669; font-weight:600;">{b_val} ({b_fmt or "Barcode"})</span>'
                else:
                    barcode_disp = '<span style="color:#64748B;">None detected</span>'

                ocr_val = selected_record.get('awb_ocr')
                ocr_src = selected_record.get('awb_ocr_source', '')
                if ocr_val:
                    ocr_disp = f'<span style="color:#4338CA; font-weight:600;">{ocr_val}</span>'
                else:
                    ocr_disp = '<span style="color:#64748B;">None extracted</span>'

                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:16px;">
                    <p style="margin-bottom:8px;"><strong>Status:</strong> <span class="badge {badge_class}">{status}</span></p>
                    <p style="margin-bottom:6px;"><strong>AWB Number:</strong> {awb_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Cross-Validation:</strong> <span class="badge {verif_class}">{verif}</span></p>
                    <p style="margin-bottom:6px;"><strong>Barcode Source:</strong> {barcode_disp}</p>
                    <p style="margin-bottom:6px;"><strong>OCR Source:</strong> {ocr_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Location:</strong> {loc_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Date:</strong> {date_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Time:</strong> {time_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Raw Timestamp:</strong> {raw_ts_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Weight:</strong> {wt_disp}</p>
                    <p style="margin-bottom:6px;"><strong>Dimensions:</strong> {dims_disp}</p>
                    <p style="margin-bottom:6px;"><strong>OCR Confidence:</strong> {selected_record['ocr_confidence']:.2f}</p>
                    <p style="margin-bottom:0px;"><strong>Audit Reason:</strong> {selected_record['reason']}</p>
                </div>
                """, unsafe_allow_html=True)

                if ocr_src:
                    st.caption(f"**OCR Evidence:** {ocr_src}")

                with st.expander("🔍 View Raw Extracted OCR Text", expanded=False):
                    if selected_record["extracted_text"]:
                        st.code(selected_record["extracted_text"], language="text")
                    else:
                        st.write("No raw text recognized.")
        else:
            st.info("No records match the current filter criteria.")


if __name__ == "__main__":
    main()
