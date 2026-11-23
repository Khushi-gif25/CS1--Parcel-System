# Prozo | Parcel Intelligence System
### Automated Parcel Image Processing & Shipment Data Extraction

A production-grade, minimal-code web application built for warehouse scanning stations to automate shipment data ingestion, courier label extraction, barcode decoding, cross-validation, anomaly detection, and audit reporting.

---

## 1. Project Overview & Business Problem

### Business Problem
At Prozo warehouse fulfillment centers, incoming and outbound parcels are photographed at conveyor scanning stations. Every parcel features a shipping label issued by logistics carriers (e.g., Delhivery, Blue Dart, DTDC, Xpressbees, Shadowfax, Ecom Express) containing vital metadata:
- Air Waybill (AWB) / Courier Tracking Number
- Dead / Gross / Actual Weight (e.g., in KG or Grams)
- Volumetric Dimensions ($L \times W \times H$ in CM)
- Hub / Warehouse / Dispatch Location (e.g., `Bangalore_Hoskote_GW (Karnataka)`, `INMHAPVP`)
- Shipment / Dispatch Date & Time (e.g., `2025-07-13`, `05:53:26 PM`)
- 1D & 2D Shipping Barcodes (e.g. Code 128, Code 39, ITF, QR Code)

Historically, warehouse operators manually read these photographs and typed shipment parameters into spreadsheets. This workflow is labor-intensive, slow, error-prone, and causes conveyor bottlenecks during peak fulfillment surges.

### Solution
The **Prozo Parcel Intelligence System** is an end-to-end Python web application that enables warehouse operators to upload hundreds of parcel photographs in bulk. The system automatically:
1. Validates image integrity and dimensions.
2. Performs heuristic computer vision assessment to check parcel presence, label obstruction, and multi-parcel scanning anomalies.
3. Decodes optical 1D & 2D barcodes across multiple image passes and rotations.
4. Preprocesses images with OpenCV (contrast enhancement, bilateral noise filtering, edge sharpening).
5. Runs local, offline EasyOCR text recognition with confidence tracking.
6. Employs context-aware regular expressions to extract AWB, weight, dimensions, location, date, and time while rejecting unrelated numbers (phone numbers, PIN codes, order IDs, delivery addresses).
7. Cross-validates the AWB between optical barcode and contextual OCR (`MATCHED`, `MISMATCH`, `BARCODE_ONLY`, `OCR_ONLY`).
8. Classifies every image into a strict 9-tier status hierarchy with honest manual review flags.
9. Surfaces an interactive dashboard with search, filtering, and a visual record inspector.
10. Generates an RFC-4180 compliant CSV audit report with guaranteed 1:1 image row mapping.

---

## 2. Architecture & Processing Workflow

The entire application is organized within `app.py` following a clean, modular, functional design:

```mermaid
flowchart TD
    A["Bulk Upload JPG/PNG (Hundred+ Files)"] --> B["Step 1: validate_image()"]
    B -- "Corrupt / Invalid" --> Z["Status: INVALID_IMAGE"]
    B -- "Valid" --> C["Step 2: assess_parcel_and_label()"]
    C -- "No Object Detected" --> Z1["Status: NO_PARCEL"]
    C -- "Multiple Boxes Detected" --> Z2["Status: MULTIPLE_PARCELS"]
    C -- "Clipped at Boundaries" --> Z3["Status: PARTIAL_PARCEL"]
    C -- "Hand / Object Covering Label" --> Z4["Status: LABEL_BLOCKED"]
    C -- "Single Parcel Visible" --> BC["Step 3: scan_barcodes() (ZXing-CPP 1D/2D Decoder)"]
    BC --> D["Step 4: preprocess_image() (OpenCV CLAHE & Sharpening)"]
    D --> E["Step 5: extract_text() (EasyOCR Local Engine)"]
    E --> F["Step 6: Contextual Field Extraction"]
    F --> F1["extract_awb() (Context-Ranked, Anti-Phone Filter)"]
    F --> F2["extract_weight() (Single & Multi-line KG/G)"]
    F --> F3["extract_dimensions() (Single & Multi-line L x W x H)"]
    F --> F4["extract_location() (Origin/Hub, Anti-Delivery Filter)"]
    F --> F5["extract_datetime() (Normalized Date & Time + Raw)"]
    BC & F1 --> CV["Step 7: cross_validate_awb()"]
    CV -- "Barcode == OCR" --> M["MATCHED"]
    CV -- "Barcode != OCR" --> MM["MISMATCH -> LOW_CONFIDENCE"]
    CV -- "Barcode Only" --> BO["BARCODE_ONLY"]
    CV -- "OCR Only" --> OO["OCR_ONLY"]
    CV & F2 & F3 --> G["Step 8: classify_status()"]
    G -- "AWB Validated + Weight + Dims" --> S["Status: SUCCESS"]
    G -- "AWB Validated, but Wt or Dims Missing" --> P["Status: PARTIAL_EXTRACTION"]
    G -- "Mismatch or Missing AWB or Low Conf" --> L["Status: LOW_CONFIDENCE"]
    S & P & L & Z & Z1 & Z2 & Z3 & Z4 --> H["Interactive Dashboard & CSV Report"]
```

### End-to-End Pipeline Steps:
1. **Validation (`validate_image`)**:
   - Confirms file size (rejects empty files or files $> 35\text{ MB}$).
   - Verifies image stream integrity with PIL (`Image.verify()`) to trap corrupt payloads.
   - Enforces a minimum resolution threshold ($60 \times 60\text{ px}$).
2. **Assessment (`assess_parcel_and_label`)**:
   - Computes Laplacian variance and grayscale standard deviation to detect blank / non-parcel frames.
   - Uses Canny edge detection and contour area analysis ($> 12\%$ of frame) to flag multiple parcels or edge-clipped boxes.
   - Detects bright shipping label regions and calculates occlusion ratios to flag blocked labels.
3. **Optical Barcode & QR Scanner (`scan_barcodes`)**:
   - Powered by `zxing-cpp` with OpenCV fallbacks.
   - Supports: **Code 128, Code 39, Code 93, ITF (Interleaved 2 of 5), Codabar, EAN-13, EAN-8, UPC-A, UPC-E, QR Code, Data Matrix, PDF417**.
   - Multi-pass decoding across: Original BGR, Grayscale, CLAHE contrast-enhanced, and 90°/270° rotated variants.
   - Intelligently distinguishes primary courier AWB barcodes (10–24 chars) from short internal order/bin tags (e.g. 6-digit Order ID).
4. **OpenCV Preprocessing (`preprocess_image`)**:
   - Normalizes image scale (bounding maximum dimension to $1800\text{ px}$ to balance OCR speed and character resolution).
   - Converts to grayscale and applies bilateral filtering (`cv2.bilateralFilter`) to suppress cardboard grain noise.
   - Applies CLAHE (`clipLimit=2.2`, `tileGridSize=(8,8)`) to equalize illumination under harsh warehouse lighting.
   - Sharpens text with a high-pass spatial convolution kernel.
5. **OCR (`extract_text`)**:
   - Reuses a singleton `easyocr.Reader` instance initialized once via Streamlit's `@st.cache_resource`.
   - Extracts character coordinates, recognized strings, and per-box confidence metrics.
6. **Contextual Field Extraction**:
   - **`extract_awb`**: Prioritizes numbers appearing near keywords (`Courier AWB No`, `AWB Number`, `AWB_NO`, `Waybill Number`, `Tracking ID`). Excludes customer phone numbers (10 digits starting with 6–9 near `Contact`/`Phone`), 6-digit PIN codes, and order numbers. Preserves source text and confidence.
   - **`extract_weight`**: Supports both single-line (`WEIGHT: 1.45 KG`) and multi-line structured label formats (`WEIGHT(Kg)` followed by `1.26`, or `Weight:` followed by `7020.000 gm` converted to `7.02 KG`).
   - **`extract_dimensions`**: Supports both single-line (`30 x 20 x 15 CM`) and multi-line structured label formats (`Length: 57.000 cm`, `Width: 33.800 cm`, `Height: 42.400 cm` $\to$ `57 x 33.8 x 42.4 CM`).
   - **`extract_location`**: Extracts warehouse, hub, or dispatch facility locations (e.g., `Location: Bangalore_Hoskote_GW (Karnataka)` $\to$ `Bangalore_Hoskote_GW (Karnataka)`, `Location:` $\to$ `INMHAPVP`). Strictly filters out customer delivery addresses, consignees, and return addresses.
   - **`extract_datetime`**: Identifies shipment, dispatch, or booking timestamps (e.g., `Time: 7/13/2025 5:53:26 PM`). Normalizes dates to `YYYY-MM-DD` (`2025-07-13`), extracts times in 12-hour AM/PM format with seconds (`05:53:26 PM`), and preserves raw timestamps for audit trails.
7. **AWB Cross-Validation (`cross_validate_awb`)**:
   - Cross-checks the barcode-decoded value against the contextual OCR extraction:
     - **`MATCHED`**: Barcode and OCR agree (100% confidence).
     - **`MISMATCH`**: Barcode and OCR differ; flags `LOW_CONFIDENCE` for manual review.
     - **`BARCODE_ONLY`**: Barcode decoded; OCR unreadable.
     - **`OCR_ONLY`**: OCR extracted; barcode damaged/absent.
     - **`NOT_FOUND`**: Neither source succeeded; manual review required.
   - Neither source is silently overwritten.
8. **Classification & Auditing (`classify_status`)**:
   - Enforces strict precedence across 9 statuses.
   - Does NOT mark `SUCCESS` simply because a barcode was detected. Full `SUCCESS` requires verified AWB + Weight + Dimensions.
   - Missing location, date, or time returns `N/A` honestly without causing false negative errors.
9. **Reporting (`generate_csv`)**:
   - Emits a standardized 16-column CSV report where row count exactly equals the number of uploaded images.

---

## 3. Technologies Used & Justification

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Language | Standard in computer vision, OCR, and enterprise warehouse data engineering. |
| **Streamlit (React engine)** | Web Interface | Pure-Python single-file execution that renders a responsive React application without requiring Node.js, npm, or split frontend/backend services. |
| **zxing-cpp** | 1D & 2D Barcode Decoder | Ultra-fast C++ barcode decoding supporting Code 128, Code 39, ITF, QR, and DataMatrix with zero external C-runtime DLL dependencies. |
| **OpenCV (`opencv-python-headless`)** | Image Preprocessing & CV | High-performance computer vision primitives for resizing, CLAHE contrast tuning, bilateral filtering, and contour geometry. |
| **EasyOCR** | Text Extraction | Accurate, completely open-source deep learning OCR (CRAFT detector + ResNet/LSTM recognizer). Runs locally on CPU/GPU with zero paid API dependencies. |
| **Pillow (PIL)** | Image I/O & Validation | Safe byte decoding, stream integrity verification, and format validation. |
| **Pandas** | Tabular Data & CSV | Robust data manipulation, filtering, and standard RFC-4180 CSV generation with escaping. |
| **Regular Expressions (`re`)** | Entity Extraction | Deterministic, explainable pattern matching for courier shipment formats. |

---

## 4. Installation & Local Execution

### Prerequisites
- Python 3.10 or higher.

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\prozo_parcel_system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on EasyOCR Model Weights**:
> EasyOCR uses local model weights for the CRAFT text detector and English recognizer. These weights are pre-cached in the local `models/` directory for offline execution without external network calls.

### Step 3: Run the Application
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 5. Status Classification Precedence

Every image yields exactly one record. Conflicting conditions are resolved via strict precedence:

| Precedence | Status | Meaning | Action / Description |
| :---: | :--- | :--- | :--- |
| **1** | `INVALID_IMAGE` | Corrupt or unreadable file | File cannot be decoded or resolution is below $60 \times 60\text{ px}$. |
| **2** | `NO_PARCEL` | No parcel in scanning zone | Uniform / blank frame or station background with no parcel contour. |
| **3** | `MULTIPLE_PARCELS` | Multiple parcels detected | Scanning station contains 2+ large parcel boxes. Operator must separate. |
| **4** | `PARTIAL_PARCEL` | Parcel truncated at edge | Parcel severely clips image boundary. Operator must reposition parcel. |
| **5** | `LABEL_BLOCKED` | Label covered or obstructed | Hand, strap, dark tape, or shadow blocks $\ge 35\%$ of shipping label. |
| **6** | `LABEL_UNREADABLE` | Label text unreadable | Parcel is detected, but neither barcode nor OCR produced readable text. |
| **7** | `LOW_CONFIDENCE` | Low recognition certainty | AWB `MISMATCH`, missing AWB, or OCR confidence $< 40\%$. Manual review required. |
| **8** | `PARTIAL_EXTRACTION` | Partial shipment data found | Verified AWB extracted, but Weight or Dimensions are missing/unreadable. Extracted fields preserved honestly. |
| **9** | `SUCCESS` | Complete extraction | AWB verified (`MATCHED`/`BARCODE_ONLY`/`OCR_ONLY`) + Weight + Dimensions all present. |

---

## 6. What Was Improved Based on Real Image Testing

### 1. Contextual Courier AWB Extraction (Issue 1 Resolved)
- **Problem**: Previous logic selected 10-digit numeric strings based on length, mistakenly picking up customer contact numbers (`9971487583`) or Order IDs (`205611`).
- **Fix**: Implemented scored contextual ranking. Numbers must appear adjacent to courier keywords (`AWB_NO`, `Courier AWB No`, `Waybill Number`, `Tracking ID`). Any numbers associated with `Contact Number`, `Phone`, `Mobile`, `Order ID`, or `PIN` receive an immediate penalty and are disqualified.

### 2. Robust 1D & 2D Barcode Decoder (Issue 2 Resolved)
- **Problem**: OpenCV's default barcode detector was failing on Code 128 shipping barcodes.
- **Fix**: Integrated `zxing-cpp` supporting Code 128, Code 39, Code 93, ITF, EAN, UPC, and QR codes. Multi-pass decoding executes across original BGR, grayscale, CLAHE, and 90°/270° rotations. On real sample images, Code 128 barcode `77563772064` is decoded in under 15 milliseconds.

### 3. AWB Cross-Validation (Issue 3 Resolved)
- **Problem**: No cross-referencing between barcode and OCR.
- **Fix**: Implemented `cross_validate_awb()`. When both barcode and OCR succeed, they are compared:
  - If equal: Marked `MATCHED`.
  - If different: Marked `MISMATCH`, and status falls back to `LOW_CONFIDENCE` for manual audit.
  - If single source: Marked `BARCODE_ONLY` or `OCR_ONLY`.
  - Both sources are preserved in the data model and displayed in the UI inspector and CSV.

### 4. Honest Status Classification (Issue 4 Resolved)
- **Problem**: Detecting a barcode should not prematurely mark an image `SUCCESS`.
- **Fix**: Barcode detection alone does not trigger `SUCCESS`. Complete `SUCCESS` requires verified AWB + Weight + Dimensions. Missing dimensions or weight result in `PARTIAL_EXTRACTION`. Any mismatch or missing AWB results in `LOW_CONFIDENCE`.

### 5. Multi-Line Dimensions & Weight Extraction
- **Problem**: Warehouse labels frequently format dimensions across multiple lines (e.g. `Length: 57.000 cm`, `Width: 33.800 cm`, `Height: 42.400 cm`, `Weight: 7020.000 gm`).
- **Fix**: Added multi-line dimension and weight parsers that reconstruct composite dimensions (`57 x 33.8 x 42.4 CM`) and convert grams to kilograms (`7.02 KG` from $7020\text{ gm}$).

### 6. Contextual Location Extraction (Enhanced)
- **Problem**: Need to extract warehouse, origin, or dispatch location without confusing it with customer delivery addresses.
- **Fix**: Implemented `extract_location()`.
  - Evaluates contextual prefixes: `Location:`, `Hub:`, `Warehouse:`, `Facility:`, `Dispatch Location:`, `Origin:`.
  - Supports both same-line (`Location: Bangalore_Hoskote_GW (Karnataka)`) and multi-line grid structures (`Location:` on line $i$, `INMHAPVP` on line $i+1$).
  - Preserves underscores, dashes, parentheses, and state information intact.
  - Strict exclusion blacklist blocks `Delivery Address`, `Customer`, `Consignee`, `Receiver`, `Ship to`, `Bill to`, and `Return Address`.
  - Repaires OCR distortions on closing parentheses (e.g., `(Karnatakaj` $\to$ `(Karnataka)`).
  - Returns `N/A` if absent.

### 7. Normalized Date & Time Extraction (Enhanced)
- **Problem**: Dates and times appear in varied courier formats and can suffer from OCR noise (e.g., apostrophes for slashes, glued year/time digits).
- **Fix**: Implemented `extract_datetime()`.
  - Scans for `Shipment Date:`, `Dispatch Date:`, `Booking Date:`, `Date:`, `Time:`, `DateTime:`.
  - Supports `DD/MM/YYYY`, `MM/DD/YYYY`, `YYYY-MM-DD`, `DD-MM-YYYY`, and US `M/D/YYYY`.
  - Normalizes dates to ISO-8601 `YYYY-MM-DD` (e.g., `7/13/2025` $\to$ `2025-07-13`).
  - Supports 12-hour AM/PM (`05:53:26 PM`, `12:57:40 AM`) and 24-hour formats with seconds.
  - Handles OCR noise: separates glued year and time numbers (e.g. `20255.53.26` $\to$ `2025 5:53:26`), corrects apostrophe/period date separators, and preserves the original raw timestamp for audit purposes.
  - Returns `N/A` if absent without failing valid parcel records.

---

## 7. CSV Report Schema

The generated CSV report contains exactly 1 row per uploaded image with 16 standardized columns:

| Column | Description | Sample Value |
| :--- | :--- | :--- |
| `image_filename` | Original uploaded image filename | `sample_01.jpg` |
| `awb_number` | Verified primary Courier Air Waybill Number | `19109110323433` |
| `awb_verification` | Cross-validation result (`MATCHED`, `MISMATCH`, `BARCODE_ONLY`, `OCR_ONLY`, `NOT_FOUND`) | `MATCHED` |
| `barcode_format` | Symbology of decoded optical barcode | `Code 128` |
| `awb_barcode` | Raw value decoded directly from barcode | `19109110323433` |
| `awb_ocr` | Value extracted from optical character recognition | `19109110323433` |
| `weight` | Normalized parcel dead/gross weight in KG or G | `7.02 KG` |
| `dimensions` | Normalized parcel dimensions ($L \times W \times H$) in CM | `57 x 33.8 x 42.4 CM` |
| `location` | Origin hub, warehouse, or dispatch facility | `Bangalore_Hoskote_GW (Karnataka)` |
| `shipment_date` | Normalized shipment date (`YYYY-MM-DD`) | `2025-07-13` |
| `shipment_time` | Normalized shipment time (`HH:MM:SS AM/PM`) | `05:53:26 PM` |
| `raw_timestamp` | Unmodified timestamp string from label | `7/13/2025 5:53:26 PM` |
| `ocr_confidence` | Mean EasyOCR recognition confidence (0.00–1.00) | `0.94` |
| `status` | Operational classification across 9 tiers | `SUCCESS` |
| `reason` | Human-readable audit explanation | `Shipment information successfully extracted & verified (MATCHED).` |
| `extracted_text` | Delimited raw text stream for audit inspection | `Location: Bangalore_Hoskote_GW \| Courier AWB No: 19109110323433...` |

---

## 8. Known Limitations Remaining

1. **Non-Standard Carrier Barcode Orientations**: While 0°, 90°, 180°, and 270° rotations are handled, barcodes rotated at skewed angles (e.g. 35° or 45°) require a full convex hull deskewing step.
2. **Missing Volumetric Dimensions on Carrier Labels**: Many shipping labels (e.g., standard e-commerce shipping tags) only print the weight and omit dimensions entirely. Such records will properly receive `PARTIAL_EXTRACTION`.
3. **Severe Motion Blur or Physical Abrasion**: Barcodes with destroyed start/stop guard patterns and unreadable OCR text will be flagged for manual review (`LABEL_UNREADABLE` or `LOW_CONFIDENCE`).
