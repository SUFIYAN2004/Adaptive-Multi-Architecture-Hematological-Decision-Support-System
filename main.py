"""
HemaVision AI — Next-Gen Medical UI & Hematology Intelligence Engine
Production-Grade Streamlit Web Application

Integrates 14-block pipeline components:
- Automated Image Quality Gating (Blur, Contrast, Staining, Noise)
- YOLOv11 Cell Localization & Detection
- U-Net Semantic Segmentation & Morphological Extraction
- Dual-Path Classifier (EfficientNetB0 vs. Vision Transformer)
- Explainable AI (Grad-CAM Integration)
- Multi-Condition Risk Analysis Engine & ISO Clinical Reporting
"""

from datetime import datetime
from pathlib import Path
import pickle
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tv_models

# ============================================================
# SYSTEM CONFIGURATION & GLOBAL CONSTANTS
# ============================================================

st.set_page_config(
    page_title="HemaVision AI | Hematology Decision Engine",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = Path(__file__).parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFICATION_CLASSES = ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil']
SEGMENTATION_CLASSES = ['background', 'cytoplasm', 'nucleus']
RISK_CONDITIONS = [
    'healthy', 'anemia', 'leukemia_suspicion', 
    'bacterial_infection', 'viral_infection', 'thrombocytopenia'
]
RISK_FEATURE_COLS = [
    'wbc', 'rbc', 'hemoglobin', 'platelets',
    'neutrophil_pct', 'lymphocyte_pct', 'monocyte_pct',
    'eosinophil_pct', 'basophil_pct', 'nc_ratio_mean'
]

REFERENCE_RANGES = {
    'wbc': (4.0, 11.0, 'x10³/µL'),
    'rbc': (4.5, 6.0, 'x10⁶/µL'),
    'hemoglobin': (13.0, 17.0, 'g/dL'),
    'platelets': (150.0, 450.0, 'x10³/µL'),
    'neutrophil_pct': (40.0, 75.0, '%'),
    'lymphocyte_pct': (20.0, 45.0, '%'),
    'monocyte_pct': (2.0, 10.0, '%'),
    'eosinophil_pct': (1.0, 6.0, '%'),
    'basophil_pct': (0.0, 2.0, '%'),
}

FRIENDLY_NAMES = {
    'wbc': 'Leukocytes (WBC)',
    'rbc': 'Erythrocytes (RBC)',
    'hemoglobin': 'Hemoglobin (Hgb)',
    'platelets': 'Thrombocytes (PLT)',
    'neutrophil_pct': 'Neutrophils',
    'lymphocyte_pct': 'Lymphocytes',
    'monocyte_pct': 'Monocytes',
    'eosinophil_pct': 'Eosinophils',
    'basophil_pct': 'Basophils',
}

# ============================================================
# CUSTOM STYLING & GLASSMORPHISM INJECTION
# ============================================================

st.markdown("""
    <style>
    /* Dark Clinical Theme Injection */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #881337 100%);
        border: 1px solid rgba(244, 63, 94, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #fda4af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.0rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Metric Glass Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(244, 63, 94, 0.4);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #f43f5e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status Badges */
    .badge-pass {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Section Cards */
    .section-box {
        background: #161b22;
        border-radius: 12px;
        border: 1px solid #30363d;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f43f5e !important;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# DEEP LEARNING ARCHITECTURES & CACHED LOADERS
# ============================================================

class RiskPredictionDNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@st.cache_resource(show_spinner=False)
def load_detection_model():
    weight_path = MODEL_DIR / "yolov11_bccd_best.pt"
    if not weight_path.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(weight_path))
    except Exception as e:
        st.error(f"Error loading YOLOv11 detector: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    weight_path = MODEL_DIR / "unet_best.pt"
    if not weight_path.exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=len(SEGMENTATION_CLASSES)
        )
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Error loading U-Net segmenter: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    weight_path = MODEL_DIR / "cnn_best.pt"
    if not weight_path.exists():
        return None
    try:
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSIFICATION_CLASSES))
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Error loading EfficientNetB0: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_vit_model():
    weight_path = MODEL_DIR / "vit_best.pt"
    if not weight_path.exists():
        return None
    try:
        import timm
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=len(CLASSIFICATION_CLASSES))
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Error loading Vision Transformer: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_risk_model():
    weight_path = MODEL_DIR / "risk_dnn_best.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not weight_path.exists() or not scaler_path.exists():
        return None, None
    try:
        model = RiskPredictionDNN(in_features=len(RISK_FEATURE_COLS), num_classes=len(RISK_CONDITIONS))
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading Risk DNN / Scaler: {e}")
        return None, None


# ============================================================
# PROCESSING PIPELINE & ANALYTICAL MODULES
# ============================================================

def assess_image_quality(image: Image.Image):
    """Quality Gating: Blur, Brightness, Contrast, Staining, and High-Frequency Noise Verification."""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = blur_score > 100.0

    brightness = float(gray.mean())
    brightness_ok = 60.0 <= brightness <= 220.0

    contrast = float(gray.std())
    contrast_ok = contrast > 20.0

    r_mean, g_mean, b_mean = img_np[..., 0].mean(), img_np[..., 1].mean(), img_np[..., 2].mean()
    color_std = float(np.std([r_mean, g_mean, b_mean]))
    staining_ok = 5.0 < color_std < 60.0

    noise_map = cv2.Laplacian(gray, cv2.CV_64F)
    noise_score = float(noise_map.std())
    noise_ok = noise_score < 40.0

    details = {
        'Focus Quality': (blur_ok, f"Var: {blur_score:.1f}"),
        'Luminance': (brightness_ok, f"Mean: {brightness:.1f}"),
        'Contrast Ratio': (contrast_ok, f"Std: {contrast:.1f}"),
        'Stain Index': (staining_ok, f"Balance: {color_std:.1f}"),
        'Signal-to-Noise': (noise_ok, f"Noise: {noise_score:.1f}"),
    }
    passed = all(v[0] for v in details.values())
    return passed, details


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


CLF_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_cell_dual_path(cnn_model, vit_model, crop_np: np.ndarray):
    if crop_np.size == 0:
        return None
    img = Image.fromarray(crop_np).convert('RGB')
    x = CLF_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    result = {}
    with torch.inference_mode():
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

    winner = max(result, key=lambda k: result[k][1])
    result['selected'] = winner
    return result


def extract_morphology(seg_model, crop_np: np.ndarray):
    if crop_np.size == 0:
        return {'nc_ratio': 0.35, 'mask': None}
    img = Image.fromarray(crop_np).convert('RGB')
    x = CLF_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        out = seg_model(x)
        mask = out.argmax(1)[0].cpu().numpy()
        
    nucleus_mask = (mask == 2).astype(np.uint8)
    cell_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    nuc_area, cell_area = nucleus_mask.sum(), cell_mask.sum()
    nc_ratio = float(nuc_area / cell_area) if cell_area > 0 else 0.35
    return {'nc_ratio': float(np.clip(nc_ratio, 0.05, 0.95)), 'mask': mask}


def generate_gradcam_heatmap(cnn_model, crop_np: np.ndarray):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        img = Image.fromarray(crop_np).convert('RGB')
        x = CLF_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
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


def predict_risk_profile(risk_model, scaler, features_dict: dict):
    x = np.array([[features_dict[c] for c in RISK_FEATURE_COLS]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(risk_model(x_tensor), dim=1).cpu().numpy()[0]
    return {RISK_CONDITIONS[i]: round(float(probs[i]) * 100, 1) for i in range(len(RISK_CONDITIONS))}


def render_annotated_image(image: Image.Image, detections: list) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    colors = {'rbc': '#10B981', 'wbc': '#3B82F6', 'platelets': '#F43F5E'}
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = colors.get(det['class'].lower(), '#F59E0B')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = det.get('wbc_subtype', det['class']).upper()
        draw.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 8 + 4, max(0, y1)], fill=color)
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill="#FFFFFF")
        
    return img_copy


def flag_abnormalities(values: dict) -> list:
    flags = []
    for key, (low, high, unit) in REFERENCE_RANGES.items():
        val = values.get(key)
        if val is None:
            continue
        name = FRIENDLY_NAMES.get(key, key)
        if val < low:
            flags.append(f"Decreased {name}: {val:.1f} {unit} (Reference Range: {low:.1f}-{high:.1f})")
        elif val > high:
            flags.append(f"Elevated {name}: {val:.1f} {unit} (Reference Range: {low:.1f}-{high:.1f})")
    return flags


def synthesize_clinical_interpretation(risk_pct: dict, abnormalities: list) -> str:
    sorted_risks = sorted(risk_pct.items(), key=lambda kv: -kv[1])
    top_cond, top_pct = sorted_risks[0]

    if top_cond == 'healthy' and top_pct >= 60:
        return "Hematological profile shows no acute pathological deviations. Morphological characteristics remain within physiological limits."

    condition_map = {
        'anemia': 'Erythrocytic indices aligned with Anemic patterns',
        'leukemia_suspicion': 'Significant blast cell count/atypical morphology requiring immediate path/hematology correlation',
        'bacterial_infection': 'Acute Leukocytosis and Neutrophilia indicative of systemic bacterial process',
        'viral_infection': 'Relative Lymphocytosis consistent with reactive viral etiology',
        'thrombocytopenia': 'Marked reduction in peripheral platelet density',
    }

    suspected = [condition_map[c] for c, p in sorted_risks if c != 'healthy' and p >= 20.0][:2]
    if not suspected:
        suspected = [condition_map.get(top_cond, top_cond.replace('_', ' '))]

    summary = f"Primary Clinical Impression: {'; '.join(suspected)}."
    if abnormalities:
        summary += f" Supporting Findings: {len(abnormalities)} quantitative marker(s) outside physiological thresholds."
    summary += " Direct correlation with bone marrow biopsy or automated flow cytometry recommended."
    return summary


# ============================================================
# USER INTERFACE CONSTRUCTION
# ============================================================

# --- HERO HEADER ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🩸 HemaVision AI Intelligence Engine</div>
        <div class="hero-subtitle">Adaptive Multi-Architecture Hematological Decision Support & Microscopic Cytometry System</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 📋 Patient Demographics")
    patient_id = st.text_input("Patient ID", value="PAT-88301")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        age = st.number_input("Age", min_value=0, max_value=120, value=42)
    with col_s2:
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    clinical_notes = st.text_area("Clinical Context", value="Routine CBC workup. Patient presents with mild fatigue.", height=80)

    st.markdown("---")
    st.markdown("### ⚙️ Deep Learning Engines")
    
    det_model = load_detection_model()
    seg_model = load_segmentation_model()
    cnn_model = load_cnn_model()
    vit_model = load_vit_model()
    risk_model, scaler = load_risk_model()

    # Fixed: Single quotes used inside HTML string inside f-string interpolation
    badge_pass = "<span class='badge-pass'>ACTIVE</span>"
    badge_warn = "<span class='badge-warn'>OFFLINE</span>"

    st.markdown(f"**YOLOv11 Detector:** {badge_pass if det_model else badge_warn}", unsafe_allow_html=True)
    st.markdown(f"**U-Net Segmenter:** {badge_pass if seg_model else badge_warn}", unsafe_allow_html=True)
    st.markdown(f"**CNN Classifier:** {badge_pass if cnn_model else badge_warn}", unsafe_allow_html=True)
    st.markdown(f"**ViT Classifier:** {badge_pass if vit_model else badge_warn}", unsafe_allow_html=True)
    st.markdown(f"**Risk Prediction DNN:** {badge_pass if risk_model else badge_warn}", unsafe_allow_html=True)

    st.markdown("---")
    conf_threshold = st.slider("Detection Confidence Cutoff", 0.10, 0.90, 0.35, 0.05)
    show_gradcam = st.checkbox("Generate Grad-CAM Visualizations", value=True)


# --- MAIN WORKFLOW & FILE INPUT ---
uploaded_file = st.file_uploader("Upload Peripheral Blood Smear Microscopic Image (JPG, PNG, TIFF)", type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Automated Quality Check Gating Block
    quality_ok, quality_details = assess_image_quality(image)

    st.markdown("### 1. Automated Microscopic Quality Audit")
    q_cols = st.columns(5)
    for i, (check_name, (ok, detail)) in enumerate(quality_details.items()):
        badge = f'<span class="badge-pass">PASS</span>' if ok else f'<span class="badge-warn">WARN</span>'
        with q_cols[i]:
            st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 0.8rem; color: #9ca3af;">{check_name}</div>
                    <div style="margin: 6px 0;">{badge}</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">{detail}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("🖼️ Raw Microscopic Field of View")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("⚡ Diagnostic Pipeline Trigger")
        st.markdown("Initiate automated YOLOv11 localization, U-Net morphological extraction, dual-path classification (CNN + ViT), and neural risk assessment.")
        analyze_clicked = st.button("🚀 Run Complete Diagnostic Suite", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        if not all([det_model, seg_model, cnn_model, risk_model]):
            st.error("Missing model weights! Please ensure trained `.pt` files reside inside `./models/` directory.")
            st.stop()

        with st.status("Executing Multi-Stage Deep Learning Pipeline...", expanded=True) as status:
            st.write("🔍 Running Cell Localization (YOLOv11)...")
            detections = run_detection(det_model, image, conf=conf_threshold)

            st.write("🧬 Running Morphological Segmentation & Dual-Path Classification...")
            wbc_subtype_counts = {c: 0 for c in CLASSIFICATION_CLASSES}
            nc_ratios = []
            model_agreement = {'agree': 0, 'disagree': 0}
            first_wbc_crop = None

            for det in detections:
                if det['class'].lower() == 'wbc':
                    clf_result = classify_cell_dual_path(cnn_model, vit_model, det['crop'])
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

                    morph = extract_morphology(seg_model, det['crop'])
                    nc_ratios.append(morph['nc_ratio'])

            status.update(label="Diagnostic Pipeline Complete!", state="complete", expanded=False)

        rbc_count = sum(1 for d in detections if d['class'].lower() == 'rbc')
        wbc_count = sum(1 for d in detections if d['class'].lower() == 'wbc')
        platelet_count = sum(1 for d in detections if d['class'].lower() == 'platelets')
        total_wbc_classified = sum(wbc_subtype_counts.values())
        mean_nc_ratio = float(np.mean(nc_ratios)) if nc_ratios else 0.35

        # --- DIAGNOSTIC TABS ---
        tab_detection, tab_classification, tab_explainability, tab_risk, tab_report = st.tabs([
            "🎯 Cell Cytometry & Detection",
            "⚔️ Dual-Path AI (CNN vs ViT)",
            "💡 Grad-CAM Visual Explainability",
            "📊 Clinical Risk DNN Engine",
            "📄 Export Diagnostic Report"
        ])

        # TAB 1: DETECTION
        with tab_detection:
            d_col1, d_col2 = st.columns([1.3, 1])
            with d_col1:
                annotated = render_annotated_image(image, detections)
                st.image(annotated, caption="YOLOv11 Bounding Box Cytometry Overlay", use_container_width=True)
            with d_col2:
                st.markdown("### Enumeration Summary")
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#10b981;">{rbc_count}</div><div class="metric-label">RBC Count</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#3b82f6;">{wbc_count}</div><div class="metric-label">WBC Count</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#f43f5e;">{platelet_count}</div><div class="metric-label">PLT Count</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Differential Leukocyte Distribution")
                if total_wbc_classified > 0:
                    diff_df = pd.DataFrame({
                        'Subtype': [c.title() for c in wbc_subtype_counts.keys()],
                        'Count': list(wbc_subtype_counts.values()),
                    })
                    fig_donut = px.pie(
                        diff_df, names='Subtype', values='Count', hole=0.6,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig_donut.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e6edf3'),
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No Leukocytes detected in this field.")

        # TAB 2: MODEL COMPARISON
        with tab_classification:
            st.markdown("### Dual-Path Model Convergence Evaluation")
            st.markdown("Simultaneous inference run across EfficientNetB0 (CNN) and Vision Transformer (ViT) architectures to verify cross-model consensus.")
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="glass-card"><div class="metric-value">{total_wbc_classified}</div><div class="metric-label">Total Evaluated</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#10b981;">{model_agreement["agree"]}</div><div class="metric-label">Consensus</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#f59e0b;">{model_agreement["disagree"]}</div><div class="metric-label">Divergence</div></div>', unsafe_allow_html=True)

            if total_wbc_classified > 0:
                agreement_rate = (model_agreement['agree'] / total_wbc_classified) * 100
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(agreement_rate / 100, text=f"Inter-Architecture Agreement Index: {agreement_rate:.1f}%")

        # TAB 3: EXPLAINABILITY
        with tab_explainability:
            st.markdown("### Gradient-Weighted Class Activation Mapping (Grad-CAM)")
            if show_gradcam and first_wbc_crop is not None:
                cam_img, gc_class, gc_conf = generate_gradcam_heatmap(cnn_model, first_wbc_crop)
                if cam_img is not None:
                    g1, g2, g3 = st.columns([1, 1, 1])
                    with g1:
                        st.markdown("#### Isolated Cell Crop")
                        st.image(first_wbc_crop, use_container_width=True)
                    with g2:
                        st.markdown("#### Grad-CAM Heatmap")
                        st.image(cam_img, use_container_width=True)
                    with g3:
                        st.markdown("#### Neural Focus Meta")
                        st.markdown(f'<div class="glass-card"><div class="metric-value">{gc_class.title()}</div><div class="metric-label">Target Subtype</div></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<div class="glass-card"><div class="metric-value" style="color:#10b981;">{gc_conf * 100:.1f}%</div><div class="metric-label">CNN Confidence</div></div>', unsafe_allow_html=True)
                else:
                    st.info("Grad-CAM generation failed. Check torch dependency.")
            else:
                st.info("Grad-CAM disabled or no WBC crop available.")

        # TAB 4: RISK EVALUATION GAUGE
        with tab_risk:
            st.markdown("### Integrated CBC Parameter Fine-Tuning")
            with st.expander("⚙️ Adjust Fine-Tuned Values", expanded=True):
                r1, r2 = st.columns(2)
                with r1:
                    wbc_abs = st.number_input("WBC Count (x10³/µL)", value=float(min(max(wbc_count * 0.8, 4.0), 40.0)), step=0.1)
                    rbc_abs = st.number_input("RBC Count (x10⁶/µL)", value=5.0, step=0.1)
                    hgb = st.number_input("Hemoglobin (g/dL)", value=14.0, step=0.1)
                    plt_abs = st.number_input("Platelet Count (x10³/µL)", value=float(min(max(platelet_count * 3, 50), 450)), step=1.0)
                with r2:
                    neu_p = st.number_input("Neutrophil %", value=float(wbc_subtype_counts['neutrophil']/max(total_wbc_classified, 1)*100) if total_wbc_classified else 58.0, step=0.1)
                    lym_p = st.number_input("Lymphocyte %", value=float(wbc_subtype_counts['lymphocyte']/max(total_wbc_classified, 1)*100) if total_wbc_classified else 30.0, step=0.1)
                    mono_p = st.number_input("Monocyte %", value=float(wbc_subtype_counts['monocyte']/max(total_wbc_classified, 1)*100) if total_wbc_classified else 6.0, step=0.1)
                    eos_p = st.number_input("Eosinophil %", value=float(wbc_subtype_counts['eosinophil']/max(total_wbc_classified, 1)*100) if total_wbc_classified else 3.0, step=0.1)
                    baso_p = st.number_input("Basophil %", value=float(wbc_subtype_counts['basophil']/max(total_wbc_classified, 1)*100) if total_wbc_classified else 0.7, step=0.1)

            risk_features = {
                'wbc': wbc_abs, 'rbc': rbc_abs, 'hemoglobin': hgb, 'platelets': plt_abs,
                'neutrophil_pct': neu_p, 'lymphocyte_pct': lym_p, 'monocyte_pct': mono_p,
                'eosinophil_pct': eos_p, 'basophil_pct': baso_p, 'nc_ratio_mean': mean_nc_ratio
            }

            risk_pct = predict_risk_profile(risk_model, scaler, risk_features)
            top_cond = max(risk_pct, key=risk_pct.get)
            healthy_p = risk_pct.get('healthy', 0)
            overall_risk_score = 100 - healthy_p

            rc1, rc2 = st.columns([1.2, 1])
            with rc1:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall_risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Pathological Risk Index", 'font': {'color': "#e6edf3"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e6edf3"},
                        'bar': {'color': "#f43f5e"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#30363d",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                            {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.3)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6edf3'))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with rc2:
                st.markdown("#### Primary Condition Vectors")
                risk_df = pd.DataFrame({
                    'Condition': [k.replace('_', ' ').title() for k in risk_pct.keys()],
                    'Probability (%)': list(risk_pct.values())
                }).sort_values('Probability (%)', ascending=True)

                fig_bar = px.bar(
                    risk_df, x='Probability (%)', y='Condition', orientation='h',
                    color='Probability (%)', color_continuous_scale='Reds'
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e6edf3'),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 5: REPORT GENERATION
        with tab_report:
            abnormalities = flag_abnormalities(risk_features)
            interpretation = synthesize_clinical_interpretation(risk_pct, abnormalities)

            st.markdown("### 📄 ISO-Standard Diagnostic Report Generator")
            
            rep_text = f"""================================================================================
                    HEMATOLOGICAL DECISION SUPPORT REPORT
================================================================================
PATIENT METADATA
--------------------------------------------------------------------------------
Patient ID        : {patient_id}
Age / Gender      : {age} Yrs / {sex}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Clinical Context  : {clinical_notes}

--------------------------------------------------------------------------------
A. QUANTITATIVE HEMOGRAM (CBC PARAMETERS)
--------------------------------------------------------------------------------
Total Leukocytes (WBC)  : {wbc_abs:6.1f} x10³/µL    [Ref: 4.0 - 11.0]
Total Erythrocytes (RBC): {rbc_abs:6.1f} x10⁶/µL    [Ref: 4.5 -  6.0]
Hemoglobin (Hgb)        : {hgb:6.1f} g/dL       [Ref: 13.0 - 17.0]
Thrombocytes (PLT)      : {plt_abs:6.0f} x10³/µL    [Ref: 150 - 450]

--------------------------------------------------------------------------------
B. DIFFERENTIAL LEUKOCYTE COUNT
--------------------------------------------------------------------------------
Neutrophils             : {neu_p:5.1f} %        [Ref: 40.0 - 75.0%]
Lymphocytes             : {lym_p:5.1f} %        [Ref: 20.0 - 45.0%]
Monocytes               : {mono_p:5.1f} %        [Ref:  2.0 - 10.0%]
Eosinophils             : {eos_p:5.1f} %        [Ref:  1.0 -  6.0%]
Basophils               : {baso_p:5.1f} %        [Ref:  0.0 -  2.0%]

--------------------------------------------------------------------------------
C. AUTOMATED ABNORMALITY DETECTION
--------------------------------------------------------------------------------
"""
            if abnormalities:
                for ab in abnormalities:
                    rep_text += f"  • [FLAG] {ab}\n"
            else:
                rep_text += "  • No parameters outside standard physiological reference ranges.\n"

            rep_text += f"""
--------------------------------------------------------------------------------
D. SYNTHESIZED CLINICAL IMPRESSION
--------------------------------------------------------------------------------
  {interpretation}

--------------------------------------------------------------------------------
E. MICROSCOPIC IMAGE ANALYTICS & RISK VECTORS
--------------------------------------------------------------------------------
Detected RBC Density     : {rbc_count} cells/FOV
Detected WBC Density     : {wbc_count} cells/FOV
Detected Platelet Density: {platelet_count} cells/FOV
Average Nucleus/Cytoplasm Ratio: {mean_nc_ratio:.3f}

Pathological Probabilities:
"""
            for cond, val in sorted(risk_pct.items(), key=lambda kv: -kv[1]):
                rep_text += f"  - {cond.replace('_', ' ').title():<25}: {val:5.1f}%\n"

            rep_text += f"""
================================================================================
STATUS / RISK STRATIFICATION: {overall_risk_score:.1f}% PATHOLOGICAL INDEX
================================================================================
DISCLAIMER: This report is generated by an artificial intelligence decision 
support pipeline for research purposes. It must be evaluated in conjunction with 
full clinical history and validated by a licensed pathologist/hematologist.
================================================================================
"""
            st.code(rep_text, language="text")

            st.download_button(
                label="💾 Export Clinical Diagnostic Summary (.TXT)",
                data=rep_text,
                file_name=f"HemaVision_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👆 Upload a high-resolution microscopic smear image to initiate the diagnostic pipeline.")