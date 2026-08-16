"""
Adaptive Multi-Architecture Hematological Decision Support System
Streamlit demo app — chains Models 1-5 into the end-to-end pipeline from the poster.
Covers poster Blocks 1-14 (Quality Assessment, CNN vs ViT comparison, Grad-CAM,
full 4-part report).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Model files expected in ./models/:
    models/yolov11_bccd_best.pt   (Model 2 - detection)
    models/unet_best.pt           (Model 3 - segmentation)
    models/cnn_best.pt            (Model 1 - CNN classifier, EfficientNetB0)
    models/vit_best.pt            (Model 1 - ViT classifier, vit_small_patch16_224)
    models/risk_dnn_best.pt       (Model 5 - risk prediction)
    models/scaler.pkl             (Model 5 - feature scaler)
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tv_models
from PIL import Image
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
import pickle
import cv2

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Hematological AI Decision Support", page_icon="🩸", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFICATION_CLASSES = ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil']
SEGMENTATION_CLASSES = ['background', 'cytoplasm', 'nucleus']
RISK_CONDITIONS = ['healthy', 'anemia', 'leukemia_suspicion', 'bacterial_infection',
                    'viral_infection', 'thrombocytopenia']
RISK_FEATURE_COLS = ['wbc', 'rbc', 'hemoglobin', 'platelets',
                      'neutrophil_pct', 'lymphocyte_pct', 'monocyte_pct',
                      'eosinophil_pct', 'basophil_pct', 'nc_ratio_mean']

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class RiskPredictionDNN(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# MODEL LOADING (cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_detection_model():
    weight_path = MODEL_DIR / "yolov11_bccd_best.pt"
    if not weight_path.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(weight_path))
    except Exception as e:
        st.warning(f"Could not load detection model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    weight_path = MODEL_DIR / "unet_best.pt"
    if not weight_path.exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                          in_channels=3, classes=len(SEGMENTATION_CLASSES))
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.warning(f"Could not load segmentation model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    weight_path = MODEL_DIR / "cnn_best.pt"
    if not weight_path.exists():
        return None
    try:
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSIFICATION_CLASSES))
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.warning(f"Could not load CNN classifier: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_vit_model():
    weight_path = MODEL_DIR / "vit_best.pt"
    if not weight_path.exists():
        return None
    try:
        import timm
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=len(CLASSIFICATION_CLASSES))
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.warning(f"Could not load ViT classifier: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_risk_model():
    weight_path = MODEL_DIR / "risk_dnn_best.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not weight_path.exists() or not scaler_path.exists():
        return None, None
    try:
        model = RiskPredictionDNN(in_features=len(RISK_FEATURE_COLS), num_classes=len(RISK_CONDITIONS))
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.warning(f"Could not load risk model: {e}")
        return None, None


# ============================================================
# BLOCK 2: IMAGE QUALITY ASSESSMENT
# ============================================================

def assess_image_quality(image: Image.Image):
    """Blur / brightness / contrast / staining / noise checks. Returns (passed, details dict)."""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Blur: variance of Laplacian — low variance = blurry
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = blur_score > 100

    # Brightness: mean pixel intensity
    brightness = gray.mean()
    brightness_ok = 60 <= brightness <= 220

    # Contrast: std of pixel intensity
    contrast = gray.std()
    contrast_ok = contrast > 20

    # Staining quality heuristic: Wright-Giemsa stained smears should show a purple/pink
    # color balance (channel means diverge from each other, unlike a washed-out grayscale-ish image)
    r_mean, g_mean, b_mean = img_np[..., 0].mean(), img_np[..., 1].mean(), img_np[..., 2].mean()
    color_std = np.std([r_mean, g_mean, b_mean])
    staining_ok = 5 < color_std < 60  # too low = grayscale/washed out, too high = artifact-heavy

    # Noise: high-frequency energy via Laplacian std deviation
    noise_map = cv2.Laplacian(gray, cv2.CV_64F)
    noise_score = noise_map.std()
    noise_ok = noise_score < 40

    details = {
        'Blur Detection': (blur_ok, f"Laplacian variance: {blur_score:.1f} ({'sharp' if blur_ok else 'blurry, retake recommended'})"),
        'Brightness Check': (brightness_ok, f"Mean intensity: {brightness:.1f} ({'normal' if brightness_ok else 'too dark/bright'})"),
        'Contrast Check': (contrast_ok, f"Std deviation: {contrast:.1f} ({'adequate' if contrast_ok else 'low contrast'})"),
        'Staining Quality': (staining_ok, f"Color balance std: {color_std:.1f} ({'acceptable' if staining_ok else 'stain artifact suspected'})"),
        'Noise Detection': (noise_ok, f"Noise score: {noise_score:.1f} ({'clean' if noise_ok else 'noisy image'})"),
    }
    passed = all(v[0] for v in details.values())
    return passed, details


# ============================================================
# PIPELINE STEPS
# ============================================================

def run_detection(det_model, image: Image.Image, conf=0.35):
    result = det_model.predict(image, conf=conf, verbose=False)[0]
    detections = []
    img_np = np.array(image)
    for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop = img_np[max(0, y1):y2, max(0, x1):x2]
        cls_name = result.names[int(cls_id)]
        detections.append({'class': cls_name, 'box': (x1, y1, x2, y2), 'crop': crop})
    return detections


CLF_TFMS = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_with_both(cnn_model, vit_model, crop_np):
    """Block 6: run CNN + ViT, return both predictions + which one wins (Model Comparison & Selection)."""
    if crop_np.size == 0:
        return None
    img = Image.fromarray(crop_np).convert('RGB')
    x = CLF_TFMS(img).unsqueeze(0).to(DEVICE)

    result = {}
    with torch.no_grad():
        if cnn_model is not None:
            probs = torch.softmax(cnn_model(x), dim=1)[0]
            idx = probs.argmax().item()
            result['cnn'] = (CLASSIFICATION_CLASSES[idx], float(probs[idx]))
        if vit_model is not None:
            probs = torch.softmax(vit_model(x), dim=1)[0]
            idx = probs.argmax().item()
            result['vit'] = (CLASSIFICATION_CLASSES[idx], float(probs[idx]))

    if not result:
        return None
    # Model Comparison & Selection: pick whichever model is more confident on this sample
    winner = max(result, key=lambda k: result[k][1])
    result['selected'] = winner
    return result


SEG_TFMS = CLF_TFMS


def segment_and_get_morphology(seg_model, crop_np):
    if crop_np.size == 0:
        return {'nc_ratio': 0.35, 'mask': None}
    img = Image.fromarray(crop_np).convert('RGB')
    x = SEG_TFMS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg_model(x)
        mask = out.argmax(1)[0].cpu().numpy()
    nucleus_mask = (mask == 2).astype(np.uint8)
    cell_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    nuc_area, cell_area = nucleus_mask.sum(), cell_mask.sum()
    nc_ratio = nuc_area / cell_area if cell_area > 0 else 0.35
    return {'nc_ratio': float(np.clip(nc_ratio, 0.05, 0.95)), 'mask': mask}


def predict_risk(risk_model, scaler, features_dict):
    x = np.array([[features_dict[c] for c in RISK_FEATURE_COLS]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(risk_model(x_tensor), dim=1).cpu().numpy()[0]
    return {RISK_CONDITIONS[i]: round(float(probs[i]) * 100, 1) for i in range(len(RISK_CONDITIONS))}


def draw_boxes_on_image(image: Image.Image, detections):
    from PIL import ImageDraw
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    colors = {'rbc': '#00C853', 'wbc': '#2979FF', 'platelets': '#FF1744'}
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = colors.get(det['class'].lower(), '#FFAB00')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = det.get('wbc_subtype', det['class'])
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return img_copy


# ============================================================
# BLOCK 11: GRAD-CAM (CNN only — ViT needs an attention-rollout reshape, CNN Grad-CAM is
# the standard, reliable choice for a live demo)
# ============================================================

def generate_gradcam(cnn_model, crop_np):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        img = Image.fromarray(crop_np).convert('RGB')
        x = CLF_TFMS(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(cnn_model(x), dim=1)[0]
            pred_idx = probs.argmax().item()

        target_layers = [cnn_model.features[-1]]
        cam = GradCAM(model=cnn_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]

        rgb_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cam_image, CLASSIFICATION_CLASSES[pred_idx], float(probs[pred_idx])
    except Exception:
        return None, None, None


# ============================================================
# BLOCK 12C/D: ABNORMALITY FLAGGING + INTERPRETATION
# ============================================================

REFERENCE_RANGES = {
    'wbc': (4.0, 11.0, 'x10^3/uL'),
    'rbc': (4.5, 6.0, 'x10^6/uL'),
    'hemoglobin': (13.0, 17.0, 'g/dL'),
    'platelets': (150.0, 450.0, 'x10^3/uL'),
    'neutrophil_pct': (40.0, 75.0, '%'),
    'lymphocyte_pct': (20.0, 45.0, '%'),
}

FRIENDLY_NAMES = {
    'wbc': 'WBC', 'rbc': 'RBC', 'hemoglobin': 'Hemoglobin', 'platelets': 'Platelets',
    'neutrophil_pct': 'Neutrophils', 'lymphocyte_pct': 'Lymphocytes',
}


def detect_abnormalities(values: dict):
    """Block 12C: flag any value outside standard reference ranges."""
    flags = []
    for key, (low, high, unit) in REFERENCE_RANGES.items():
        val = values.get(key)
        if val is None:
            continue
        name = FRIENDLY_NAMES[key]
        if val < low:
            flags.append(f"Low {name} ({val:.1f} {unit}, ref {low}-{high})")
        elif val > high:
            flags.append(f"High {name} ({val:.1f} {unit}, ref {low}-{high})")
    return flags


def generate_interpretation(risk_pct: dict, abnormalities: list):
    """Block 12D: plain-English interpretation sentence, generated from top risk conditions."""
    sorted_risks = sorted(risk_pct.items(), key=lambda kv: -kv[1])
    top_cond, top_pct = sorted_risks[0]

    if top_cond == 'healthy' and top_pct >= 60:
        return "Findings are within expected normal limits. No significant abnormalities detected. Routine follow-up recommended."

    condition_phrases = {
        'anemia': 'anemia',
        'leukemia_suspicion': 'possible leukemic process warranting hematologist review',
        'bacterial_infection': 'bacterial infection',
        'viral_infection': 'viral infection',
        'thrombocytopenia': 'thrombocytopenia',
    }

    mentioned = [condition_phrases[c] for c, p in sorted_risks if c != 'healthy' and p >= 20][:3]
    if not mentioned:
        mentioned = [condition_phrases.get(top_cond, top_cond.replace('_', ' '))]

    sentence = f"Findings suggest possible {', '.join(mentioned)}."
    if abnormalities:
        sentence += f" Supporting abnormalities: {'; '.join(abnormalities)}."
    sentence += " Clinical correlation is recommended."
    return sentence


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🩸 Adaptive Multi-Architecture Hematological Decision Support System")
st.caption("For multi-disease blood smear analysis using deep learning and explainable AI — MSc Biotechnology, VIT")

with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID", value="12345")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
    with col_b:
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    clinical_notes = st.text_area("Clinical Notes (optional)", height=80)

    st.divider()
    st.header("Model Status")
    det_model = load_detection_model()
    seg_model = load_segmentation_model()
    cnn_model = load_cnn_model()
    vit_model = load_vit_model()
    risk_model, scaler = load_risk_model()

    st.write("✅ Detection (YOLOv11)" if det_model else "❌ Detection — missing `yolov11_bccd_best.pt`")
    st.write("✅ Segmentation (U-Net)" if seg_model else "❌ Segmentation — missing `unet_best.pt`")
    st.write("✅ Classification — CNN" if cnn_model else "❌ CNN — missing `cnn_best.pt`")
    st.write("✅ Classification — ViT" if vit_model else "⚠️ ViT — missing `vit_best.pt` (CNN-only mode)")
    st.write("✅ Risk Prediction (DNN)" if risk_model else "❌ Risk Prediction — missing `risk_dnn_best.pt`/`scaler.pkl`")

    st.divider()
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
    show_gradcam = st.checkbox("Show Grad-CAM explainability", value=True)

uploaded_file = st.file_uploader("Upload Blood Smear Image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # ---- Block 2: Quality Assessment (gate before anything else runs) ----
    st.subheader("2. Image Quality Assessment")
    quality_ok, quality_details = assess_image_quality(image)

    qcols = st.columns(5)
    for i, (check_name, (ok, detail)) in enumerate(quality_details.items()):
        with qcols[i]:
            st.metric(check_name, "✅ OK" if ok else "⚠️ Issue")
            st.caption(detail)

    if quality_ok:
        st.success("QUALITY OK — proceeding with analysis")
    else:
        st.error("POOR QUALITY — Request New Image. Analysis may still be run below, but results are less reliable on this image.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Input Image")
        st.image(image, use_container_width=True)

    analyze_clicked = st.button("🔬 Analyze Smear", type="primary", use_container_width=True)

    if analyze_clicked:
        if not all([det_model, seg_model, cnn_model, risk_model]):
            st.error("One or more required models failed to load — check sidebar status.")
            st.stop()

        with st.spinner("Detecting cells (YOLOv11)..."):
            detections = run_detection(det_model, image, conf=conf_threshold)

        wbc_subtype_counts = {c: 0 for c in CLASSIFICATION_CLASSES}
        nc_ratios = []
        model_agreement = {'agree': 0, 'disagree': 0}
        first_wbc_crop = None

        with st.spinner("Classifying WBC subtypes (CNN + ViT comparison) + extracting morphology..."):
            for det in detections:
                if det['class'].lower() == 'wbc':
                    clf_result = classify_with_both(cnn_model, vit_model, det['crop'])
                    if clf_result is None:
                        continue
                    if first_wbc_crop is None:
                        first_wbc_crop = det['crop']

                    winner = clf_result['selected']
                    subtype, conf = clf_result[winner]
                    det['wbc_subtype'] = subtype
                    det['wbc_conf'] = conf
                    det['model_used'] = winner
                    wbc_subtype_counts[subtype] += 1

                    if 'cnn' in clf_result and 'vit' in clf_result:
                        if clf_result['cnn'][0] == clf_result['vit'][0]:
                            model_agreement['agree'] += 1
                        else:
                            model_agreement['disagree'] += 1

                    morph = segment_and_get_morphology(seg_model, det['crop'])
                    nc_ratios.append(morph['nc_ratio'])

        with col2:
            st.subheader("4. Cell Detection")
            annotated = draw_boxes_on_image(image, detections)
            st.image(annotated, use_container_width=True)

        rbc_count = sum(1 for d in detections if d['class'].lower() == 'rbc')
        wbc_count = sum(1 for d in detections if d['class'].lower() == 'wbc')
        platelet_count = sum(1 for d in detections if d['class'].lower() == 'platelets')
        total_wbc_classified = sum(wbc_subtype_counts.values())

        # ---- Block 6: Model Comparison & Selection summary ----
        st.divider()
        st.subheader("6. Cell Classification — CNN vs ViT Comparison")
        if vit_model is not None and total_wbc_classified > 0:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Cells classified", total_wbc_classified)
            mc2.metric("Models agreed", model_agreement['agree'])
            mc3.metric("Models disagreed", model_agreement['disagree'])
            st.caption("Per cell, the higher-confidence model's prediction is used (Model Comparison & Selection, per poster Block 6).")
        elif total_wbc_classified > 0:
            st.info(f"ViT checkpoint not found — ran CNN only on {total_wbc_classified} WBCs.")
        else:
            st.info("No WBCs detected to classify.")

        # ---- Block 11: Grad-CAM ----
        if show_gradcam and first_wbc_crop is not None:
            st.divider()
            st.subheader("11. Explainable AI — Grad-CAM")
            cam_image, gc_class, gc_conf = generate_gradcam(cnn_model, first_wbc_crop)
            if cam_image is not None:
                gc1, gc2, gc3 = st.columns([1, 1, 1])
                with gc1:
                    st.image(first_wbc_crop, caption="Original Cell", use_container_width=True)
                with gc2:
                    st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                with gc3:
                    st.metric("Confidence Score", f"{gc_conf*100:.1f}%")
                    st.metric("Predicted Class", gc_class.title())
            else:
                st.info("Grad-CAM unavailable — install `grad-cam` package (`pip install grad-cam`).")

        st.divider()
        st.subheader("8. Cell Counting")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("RBC Count", rbc_count)
        cc2.metric("WBC Count", wbc_count)
        cc3.metric("Platelet Count", platelet_count)

        if total_wbc_classified > 0:
            st.markdown("**Differential WBC Count**")
            diff_df = pd.DataFrame({
                'Cell Type': list(wbc_subtype_counts.keys()),
                'Count': list(wbc_subtype_counts.values()),
                'Percent': [round(v / total_wbc_classified * 100, 1) for v in wbc_subtype_counts.values()]
            })
            fig_pie = px.pie(diff_df, names='Cell Type', values='Count', hole=0.45,
                              title="Differential WBC Count (%)")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.subheader("10. Multi-Disease Risk Prediction")
        mean_nc_ratio = float(np.mean(nc_ratios)) if nc_ratios else 0.35

        with st.expander("⚙️ Adjust CBC values (image-derived values pre-filled where possible)", expanded=True):
            r1, r2 = st.columns(2)
            with r1:
                wbc_abs = st.number_input("WBC (x10³/µL)", value=float(min(max(wbc_count * 0.8, 4.0), 40.0)), step=0.1)
                rbc_abs = st.number_input("RBC (x10⁶/µL)", value=5.0, step=0.1)
                hgb = st.number_input("Hemoglobin (g/dL)", value=14.0, step=0.1)
                plt_abs = st.number_input("Platelets (x10³/µL)", value=float(min(max(platelet_count * 3, 50), 450)), step=1.0)
            with r2:
                neu_pct = st.number_input("Neutrophil %", value=float(wbc_subtype_counts['neutrophil']/max(total_wbc_classified,1)*100) if total_wbc_classified else 58.0, step=0.1)
                lym_pct = st.number_input("Lymphocyte %", value=float(wbc_subtype_counts['lymphocyte']/max(total_wbc_classified,1)*100) if total_wbc_classified else 30.0, step=0.1)
                mono_pct = st.number_input("Monocyte %", value=float(wbc_subtype_counts['monocyte']/max(total_wbc_classified,1)*100) if total_wbc_classified else 6.0, step=0.1)
                eos_pct = st.number_input("Eosinophil %", value=float(wbc_subtype_counts['eosinophil']/max(total_wbc_classified,1)*100) if total_wbc_classified else 3.0, step=0.1)
                baso_pct = st.number_input("Basophil %", value=float(wbc_subtype_counts['basophil']/max(total_wbc_classified,1)*100) if total_wbc_classified else 0.7, step=0.1)

        risk_features = {
            'wbc': wbc_abs, 'rbc': rbc_abs, 'hemoglobin': hgb, 'platelets': plt_abs,
            'neutrophil_pct': neu_pct, 'lymphocyte_pct': lym_pct, 'monocyte_pct': mono_pct,
            'eosinophil_pct': eos_pct, 'basophil_pct': baso_pct, 'nc_ratio_mean': mean_nc_ratio,
        }

        risk_pct = predict_risk(risk_model, scaler, risk_features)
        top_condition = max(risk_pct, key=risk_pct.get)
        healthy_pct = risk_pct.get('healthy', 0)
        overall_risk = "LOW" if healthy_pct >= 70 else ("MODERATE" if healthy_pct >= 40 else "HIGH")
        risk_color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red"}[overall_risk]

        rc1, rc2 = st.columns([1.2, 1])
        with rc1:
            risk_df = pd.DataFrame({
                'Condition': [k.replace('_', ' ').title() for k in risk_pct.keys()],
                'Risk %': list(risk_pct.values())
            }).sort_values('Risk %', ascending=True)
            fig_bar = px.bar(risk_df, x='Risk %', y='Condition', orientation='h',
                              color='Risk %', color_continuous_scale='RdYlGn_r',
                              title="Predicted Risk by Condition")
            st.plotly_chart(fig_bar, use_container_width=True)
        with rc2:
            st.markdown(f"### Overall Risk Level: :{risk_color}[{overall_risk}]")
            st.markdown(f"**Top concern:** {top_condition.replace('_', ' ').title()} ({risk_pct[top_condition]}%)")
            st.markdown(f"**AI Confidence:** {max(risk_pct.values())}%")
            st.markdown(f"**Recommendation:** {'Consult Hematologist' if overall_risk != 'LOW' else 'Routine follow-up'}")

        # ---- Block 12: Full report (A/B/C/D) ----
        st.divider()
        st.subheader("12. Automated Hematological Report")

        abnormalities = detect_abnormalities(risk_features)
        interpretation = generate_interpretation(risk_pct, abnormalities)

        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown("**C. Detected Abnormalities**")
            if abnormalities:
                for a in abnormalities:
                    st.warning(f"⚠️ {a}")
            else:
                st.success("✅ No abnormalities flagged against reference ranges.")
        with rep_col2:
            st.markdown("**D. Interpretation**")
            st.info(interpretation)

        report_text = f"""HEMATOLOGICAL ANALYSIS REPORT
{'='*55}
Patient ID : {patient_id}
Age/Sex    : {age} / {sex}
Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*55}

A. CBC SUMMARY
  WBC          : {wbc_abs:.1f} x10^3/uL   (ref: 4-11)
  RBC          : {rbc_abs:.1f} x10^6/uL   (ref: 4.5-6.0)
  Hemoglobin   : {hgb:.1f} g/dL           (ref: 13-17)
  Platelets    : {plt_abs:.0f} x10^3/uL   (ref: 150-450)

B. DIFFERENTIAL COUNT
  Neutrophils  : {neu_pct:.1f}%
  Lymphocytes  : {lym_pct:.1f}%
  Monocytes    : {mono_pct:.1f}%
  Eosinophils  : {eos_pct:.1f}%
  Basophils    : {baso_pct:.1f}%

C. DETECTED ABNORMALITIES
"""
        report_text += ("\n".join(f"  - {a}" for a in abnormalities) if abnormalities else "  None detected.")
        report_text += f"""

D. INTERPRETATION
  {interpretation}

E. IMAGE-DERIVED CELL COUNTS
  RBC detected      : {rbc_count}
  WBC detected      : {wbc_count}
  Platelets detected: {platelet_count}
  Mean N/C ratio     : {mean_nc_ratio:.3f}

F. PREDICTED RISK
"""
        for cond, pct in sorted(risk_pct.items(), key=lambda kv: -kv[1]):
            report_text += f"  {cond.replace('_',' ').title():<22}: {pct:>5.1f}%\n"

        report_text += f"""
{'='*55}
Overall Risk Level : {overall_risk}
AI Confidence       : {max(risk_pct.values())}%
Recommendation      : {'Consult Hematologist' if overall_risk != 'LOW' else 'Routine follow-up'}
{'='*55}

Clinical Notes: {clinical_notes if clinical_notes else 'None provided'}

DISCLAIMER: This is an AI-assisted decision support output for research/
demonstration purposes (MSc Biotechnology project). It is NOT a substitute
for professional medical diagnosis. All findings require clinical correlation
and confirmation by a qualified hematologist/pathologist.
"""
        st.text(report_text)
        st.download_button("📄 Download Report (.txt)", data=report_text,
                            file_name=f"hematology_report_{patient_id}.txt",
                            mime="text/plain", use_container_width=True)

else:
    st.info("👆 Upload a blood smear image to begin analysis.")
    st.divider()
    st.subheader("About This System")
    st.markdown("""
    This demo covers the full 14-block pipeline from the project poster:

    **2. Quality Assessment** — blur, brightness, contrast, staining, noise checks run
    automatically on upload, before any model inference.

    **6. Classification (Model Comparison & Selection)** — every detected WBC is run
    through *both* the CNN (EfficientNetB0) and ViT (vit_small_patch16_224) classifiers;
    the higher-confidence prediction is used, and agreement/disagreement is reported.

    **11. Explainable AI** — Grad-CAM heatmap on a representative classified cell, showing
    which regions drove the CNN's prediction.

    **12. Report** — all four sections: CBC Summary, Differential Count, Detected
    Abnormalities (rule-based flagging against standard reference ranges), and a generated
    plain-English Interpretation.

    **Note:** Risk prediction (Model 5) trains on medically-informed *synthetic* patient
    data, since no public dataset pairs smear images with confirmed diagnostic outcomes.
    Real deployment requires IRB-approved validation data — stated here as a limitation,
    consistent with the project's research/proof-of-concept scope.
    """)
