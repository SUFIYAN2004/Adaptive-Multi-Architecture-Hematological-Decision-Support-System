# Hematological AI Decision Support — Streamlit Demo

Chains your 5 trained models (from the Colab notebooks) into one end-to-end web app,
matching Blocks 1–14 of the poster.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy your trained checkpoints into `models/`**

   From your Colab download zips, extract and rename/copy these files into this folder:

   | From zip | Copy to |
   |---|---|
   | `model2_cell_detection.zip` → `best.pt` | `models/yolov11_bccd_best.pt` |
   | `model3_cell_segmentation.zip` → `unet_best.pt` | `models/unet_best.pt` |
   | Model 1 notebook → `cnn_best.pt` | `models/cnn_best.pt` |
   | Model 1 notebook → `vit_best.pt` | `models/vit_best.pt` (optional — enables CNN vs ViT comparison, Block 6) |
   | `model5_risk_prediction.zip` → `risk_dnn_best.pt` | `models/risk_dnn_best.pt` |
   | `model5_risk_prediction.zip` → `scaler.pkl` | `models/scaler.pkl` |

   Final structure:
   ```
   streamlit_app/
     app.py
     requirements.txt
     models/
       yolov11_bccd_best.pt
       unet_best.pt
       cnn_best.pt
       vit_best.pt        (optional)
       risk_dnn_best.pt
       scaler.pkl
   ```

   `vit_best.pt` is optional — the app runs CNN-only if it's missing (sidebar shows a
   warning, not an error). Include it to get the CNN-vs-ViT agreement comparison in Block 6.

3. **Run**
   ```bash
   streamlit run app.py
   ```

   Opens at `http://localhost:8501`. If a model file is missing, the sidebar will show
   which one — the app still loads, it just disables analysis until all 4 are present
   (Model 4's GAN generators aren't needed at inference time, only during training).

## What it does

Upload a blood smear image → the app runs the full poster pipeline:

1. **Block 2 — Quality Assessment**: blur, brightness, contrast, staining, and noise checks
   run automatically on upload, before any model touches the image. Shows QUALITY OK /
   POOR QUALITY, same as the poster.
2. **Block 4 — YOLOv11 detection**: boxes for every RBC / WBC / Platelet
3. **Block 5 — U-Net segmentation**: nucleus/cytoplasm masks on WBC crops → N/C ratio
4. **Block 6 — CNN vs ViT classification**: every detected WBC is classified by *both*
   models; the higher-confidence prediction is used per cell, and agreement/disagreement
   across the two architectures is reported (Model Comparison & Selection)
5. **Block 8 — Cell counting**: totals + differential percentages, interactive pie chart
6. **Block 10 — DNN risk prediction**: multi-disease risk scores, editable CBC fields (since
   absolute hemoglobin/platelet concentration needs a hematology analyzer, not image analysis
   alone — the app pre-fills sensible estimates from detected counts, but you can override them)
7. **Block 11 — Grad-CAM**: heatmap on a representative classified cell, showing which
   regions drove the CNN's prediction
8. **Block 12 — Automated report**: all four sections — CBC Summary, Differential Count,
   Detected Abnormalities (rule-based flagging against reference ranges), and a generated
   plain-English Interpretation — downloadable as text

## Deploying beyond localhost

- **Streamlit Community Cloud**: push this folder to a GitHub repo, connect it at
  share.streamlit.io. Note the model files may be large — use Git LFS or host them
  externally (e.g. Google Drive / HuggingFace Hub) and download at app startup instead
  of committing large `.pt` files to the repo.
- **Cloud/Edge deployment** (per poster Block 14): this same `app.py` runs unchanged on
  a cloud VM (AWS/GCP) or edge device (Jetson) — the poster's "Deployment" block is
  satisfied by containerizing this with Docker and exposing port 8501.

## Limitations to state in your paper

- Model 5 (risk prediction) trains on medically-informed **synthetic** patient records
  (CBC values generated from standard reference intervals with condition-specific rules),
  since no public dataset pairs smear images with confirmed diagnostic outcomes. Real
  deployment requires IRB-approved patient data for validation.
- Absolute CBC values (hemoglobin g/dL, precise WBC/RBC counts per µL) require a
  hematology analyzer — image-based cell counts alone give proportions and relative
  counts, not calibrated absolute concentrations. The UI exposes these as editable
  fields for this reason.
