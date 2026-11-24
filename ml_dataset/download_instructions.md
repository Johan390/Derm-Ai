# DermAI Dataset Guide & Integration Instructions

## Target Disease Categories
DermAI targets the 7 canonical diagnostic categories established by the **HAM10000** ("Human Against Machine with 10000 training images") and **ISIC** benchmark:

1. **MEL (Melanoma)**: Malignant neoplasm of melanocytes. *High Risk / Urgent Clinical Review*
2. **NV (Melanocytic Nevi)**: Common benign mole. *Low Risk*
3. **BCC (Basal Cell Carcinoma)**: Malignant, common non-melanoma skin cancer. *High Risk / Clinical Review*
4. **AKIEC (Actinic Keratosis & Intraepithelial Carcinoma / Bowen's)**: Pre-malignant / early squamous cell carcinoma. *Moderate Risk*
5. **BKL (Benign Keratosis)**: Seborrheic keratosis, solar lentigo, and lichen-planus like keratosis. *Low Risk*
6. **DF (Dermatofibroma)**: Benign cutaneous lesion. *Low Risk*
7. **VASC (Vascular Lesions)**: Hemangiomas, angiokeratomas, pyogenic granulomas. *Low Risk*

---

## Current Workspace Status: Development Benchmark Sample
* **Current Status:** Genuine HAM10000 image archives (~2.6 GB) are **not included** in the source repository to keep the codebase lightweight and compliant with data distribution terms.
* **Sample Metadata:** The file `ml_dataset/sample_metadata.csv` contains 1,200 synthetic/sample records created for local development and workflow validation. It reproduces the published class distribution but does not contain real patient images.
* **Empirical Dataset Preparation:** Pending manual acquisition following the steps below.

---

## How to Ingest the Genuine HAM10000 Dataset

### Option A: Via Kaggle CLI (Recommended)
1. Ensure your Kaggle API token is placed at `~/.kaggle/kaggle.json`.
2. Run in terminal:
```bash
mkdir -p ml_dataset/raw_ham10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d ./ml_dataset/raw_ham10000/
```

### Option B: Via Direct Web Download
1. Visit [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Download and extract:
   - `HAM10000_images_part_1` and `HAM10000_images_part_2` (merge into `./ml_dataset/raw_ham10000/images/`)
   - `HAM10000_metadata.csv` (place into `./ml_dataset/raw_ham10000/HAM10000_metadata.csv`)

---

## Running Dataset Validation and EDA
Once the real dataset is placed into `ml_dataset/raw_ham10000/`, run:
```bash
python ml_dataset/eda_analysis.py
```
The script will automatically:
1. Detect genuine HAM10000 metadata and image folders.
2. Verify dataset integrity (missing images, duplicate IDs, invalid disease labels).
3. Compute empirical statistical distributions and update `eda_summary.json`.
4. If no genuine dataset is present, it safely analyzes `sample_metadata.csv` and labels the provenance as `SYNTHETIC_DEVELOPMENT_SAMPLE`.
