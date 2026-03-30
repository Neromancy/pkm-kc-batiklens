#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatikLens AI - Prototipe PKM-KC 2026
Identifikasi Motif Batik dengan Deep Learning (ResNet-18)
"""

import json
import os

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="BatikLens AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "# BatikLens AI 🔍\nPrototipe PKM-KC 2026\n\nModel: ResNet-18 Fine-Tuned"
    },
)

# =============================================================================
# STYLING CSS (Konsisten: Dark BG = Light Text, Light BG = Dark Text)
# =============================================================================
def load_css():
    """Memuat custom CSS untuk styling konsisten"""
    st.markdown(
        """
        <style>
            /* ========== COLOR VARIABLES ========== */
            :root {
                --color-dark: #3E333F;
                --color-brown: #5D4E60;
                --color-red: #FC4442;
                --color-yellow: #F0E19E;
                --color-light: #F2F2F2;
                --color-white: #FFFFFF;
                --color-dark-bg: #2B2D31;
            }

            /* ========== GLOBAL SETTINGS ========== */
            .stApp { background-color: var(--color-light) !important; }
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                max-width: 600px;
            }

            /* ========== BASE TEXT COLORS ========== */
            h1, h2, h3, h4, h5, h6 { color: var(--color-dark) !important; }
            p, span, label, div { color: var(--color-dark) !important; }

            /* ========== 1. CAMERA PERMISSION - BRIGHT TEXT ========== */
            [data-testid="stCameraContainer"],
            [data-testid="stCameraContainer"] *,
            [data-testid="stCameraPermission"],
            [data-testid="stCameraPermission"] * {
                color: var(--color-white) !important;
            }
            [data-testid="stCameraContainer"] svg {
                fill: var(--color-white) !important;
            }
            [data-testid="stCameraPermission"] a {
                color: #64B5F6 !important;
                text-decoration: underline !important;
            }

            /* ========== 2. FILE UPLOADER - Complete Fix ========== */
            [data-testid="stFileUploader"] {
                background: var(--color-white);
                border-radius: 12px;
                padding: 1rem;
                box-shadow: 0 2px 8px rgba(62, 51, 63, 0.1);
            }
            
            /* Label di atas uploader - DARK TEXT */
            [data-testid="stFileUploader"] > div:first-child > label,
            [data-testid="stFileUploader"] > div:first-child > label * {
                color: var(--color-dark) !important;
                font-weight: 600 !important;
            }
            
            /* Drop zone - DARK BG, WHITE TEXT */
            [data-testid="stFileUploader"] section {
                background-color: var(--color-dark-bg) !important;
                border-radius: 10px !important;
                border: 2px dashed rgba(255,255,255,0.3) !important;
            }
            [data-testid="stFileUploader"] section > div,
            [data-testid="stFileUploader"] section > div > div,
            [data-testid="stFileUploader"] section p,
            [data-testid="stFileUploader"] section span,
            [data-testid="stFileUploader"] section label { 
                color: var(--color-white) !important; 
            }
            
            /* Uploaded file container - WHITE BG */
            [data-testid="stFileUploader"] > div:nth-child(2) {
                background-color: var(--color-white) !important;
                border-radius: 8px;
                padding: 0.5rem;
                margin-top: 0.5rem;
            }
            
            /* FILENAME - DARK BROWN */
            [data-testid="stFileUploader"] > div:nth-child(2) > div,
            [data-testid="stFileUploader"] > div:nth-child(2) span,
            [data-testid="stFileUploader"] > div:nth-child(2) div > span:first-child {
                color: var(--color-brown) !important;
                font-weight: 600 !important;
            }

            /* ========== 3. TITLE & SUBTITLE ========== */
            h1 {
                text-align: center;
                font-size: 2rem !important;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: var(--color-brown) !important;
                text-align: center;
                margin-bottom: 1.5rem;
                font-size: 1rem;
                font-weight: 500;
            }

            /* ========== 4. TABS - ROUNDED & PROPER CONTRAST ========== */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.25rem !important;
                background: var(--color-white) !important;
                border-radius: 50px !important;
                padding: 0.35rem !important;
                box-shadow: 0 2px 8px rgba(62, 51, 63, 0.15) !important;
                display: inline-flex !important;
                width: auto !important;
                margin: 0 auto 1.5rem auto !important;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 50px !important;
                padding: 0.5rem 1.25rem !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
                transition: all 0.3s ease !important;
                border: none !important;
            }
            /* INACTIVE TAB - Light BG, Dark Text */
            .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
                background: var(--color-light) !important;
                color: var(--color-dark) !important;
            }
            .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) * {
                color: var(--color-dark) !important;
                fill: var(--color-dark) !important;
            }
            /* ACTIVE TAB - Dark BG, White Text */
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background: var(--color-dark) !important;
                color: var(--color-white) !important;
                box-shadow: 0 2px 6px rgba(62, 51, 63, 0.3) !important;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] * {
                color: var(--color-white) !important;
                fill: var(--color-white) !important;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] svg {
                fill: var(--color-white) !important;
            }

            /* ========== 5. EXPANDER - DARK HEADER = WHITE TEXT ========== */
            /* Closed expander header - DARK BG, WHITE TEXT */
            [data-testid="stExpander"] details:not([open]) summary {
                background-color: var(--color-dark) !important;
                color: var(--color-white) !important;
            }
            [data-testid="stExpander"] details:not([open]) summary * {
                color: var(--color-white) !important;
            }
            [data-testid="stExpander"] details:not([open]) summary svg {
                fill: var(--color-white) !important;
            }
            
            /* Open expander header - DARK BG, WHITE TEXT */
            [data-testid="stExpander"] details[open] summary {
                background-color: var(--color-dark) !important;
                color: var(--color-white) !important;
            }
            [data-testid="stExpander"] details[open] summary * {
                color: var(--color-white) !important;
            }
            [data-testid="stExpander"] details[open] summary svg {
                fill: var(--color-white) !important;
            }
            
            /* Expander content - WHITE BG, DARK BROWN TEXT */
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] {
                background-color: var(--color-white) !important;
            }
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] *,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] p,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] span,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] strong,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] li,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] h1,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] h2,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] h3,
            [data-testid="stExpander"] div[data-testid="stExpanderContent"] h4 {
                color: var(--color-brown) !important;
            }

            /* ========== 6. BUTTONS - WHITE TEXT ========== */
            /* All buttons - base style */
            .stButton > button,
            .stButton button,
            button[kind="primary"],
            button[kind="secondary"],
            button[kind="tertiary"],
            .stFormSubmitButton button,
            div.stButton > button:first-child {
                background-color: var(--color-dark) !important;
                color: var(--color-white) !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
                width: 100%;
            }
            
            /* Button text content */
            .stButton > button *,
            .stButton button *,
            button[kind="primary"] *,
            button[kind="secondary"] *,
            div.stButton > button:first-child * {
                color: var(--color-white) !important;
            }
            
            /* Button hover state */
            .stButton > button:hover,
            .stButton button:hover,
            button[kind="primary"]:hover,
            div.stButton > button:first-child:hover {
                background-color: var(--color-red) !important;
                color: var(--color-white) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(252, 68, 66, 0.3) !important;
            }
            
            .stButton > button:hover *,
            .stButton button:hover *,
            button[kind="primary"]:hover * {
                color: var(--color-white) !important;
            }
            
            /* Button focus/active state */
            .stButton > button:active,
            .stButton > button:focus,
            button[kind="primary"]:active,
            button[kind="primary"]:focus {
                background-color: var(--color-dark) !important;
                color: var(--color-white) !important;
            }

            /* ========== 7. ALERT BOXES ========== */
            .stAlert {
                border-radius: 12px !important;
                border: none !important;
                background-color: var(--color-white) !important;
            }
            .stAlert * { color: var(--color-dark) !important; }
            .stInfo {
                background-color: rgba(240, 225, 158, 0.4) !important;
                border-left: 4px solid var(--color-yellow) !important;
            }
            .stInfo *, .stInfo p, .stInfo span {
                color: var(--color-brown) !important;
            }
            .stSuccess {
                background-color: rgba(242, 242, 242, 0.9) !important;
                border-left: 4px solid var(--color-dark) !important;
            }
            .stWarning {
                background-color: rgba(240, 225, 158, 0.4) !important;
                border-left: 4px solid var(--color-yellow) !important;
            }

            /* ========== 8. PREDICTION CARDS ========== */
            .prediction-card {
                background: var(--color-white);
                border-radius: 16px;
                padding: 1.25rem;
                margin: 0.75rem 0;
                box-shadow: 0 4px 12px rgba(62, 51, 63, 0.15);
                border-left: 5px solid var(--color-dark);
            }
            .prediction-card.high-confidence { border-left-color: var(--color-red); }
            .prediction-card.medium-confidence { border-left-color: var(--color-yellow); }
            .prediction-card h4 {
                color: var(--color-dark) !important;
                margin: 0 0 0.75rem 0;
                font-size: 1.25rem;
                font-weight: 600;
            }
            .confidence-label {
                color: var(--color-brown) !important;
                font-size: 0.9rem;
                font-weight: 500;
            }
            .confidence-value {
                color: var(--color-dark) !important;
                font-weight: 700;
                font-size: 1rem;
            }
            .confidence-bar {
                height: 10px;
                background: #E8E4E8;
                border-radius: 6px;
                margin: 0.5rem 0;
                overflow: hidden;
            }
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--color-red), var(--color-yellow));
                border-radius: 6px;
                transition: width 0.3s ease;
            }
            .prediction-card p {
                color: var(--color-brown) !important;
                margin: 0.5rem 0 0 0;
                font-size: 0.9rem;
                line-height: 1.5;
            }
            .prediction-card strong { color: var(--color-dark) !important; }

            /* ========== 9. IMAGE CONTAINER ========== */
            .image-container {
                background: var(--color-white);
                border-radius: 16px;
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(62, 51, 63, 0.15);
            }

            /* ========== 10. FOOTER ========== */
            footer {
                background-color: transparent !important;
                color: var(--color-brown) !important;
            }
            footer * { color: var(--color-brown) !important; }

            /* ========== 11. MOBILE OPTIMIZATION ========== */
            @media (max-width: 768px) {
                h1 { font-size: 1.5rem !important; }
                .subtitle { font-size: 0.9rem !important; }
                .prediction-card { padding: 1rem !important; }
                .stTabs [data-baseweb="tab"] {
                    padding: 0.4rem 1rem !important;
                    font-size: 0.85rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

load_css()

# =============================================================================
# HEADER
# =============================================================================
st.markdown("<h1>🔍 BatikLens AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Identifikasi Motif Batik dengan Deep Learning • ResNet-18</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# LOAD DATABASE (Cached)
# =============================================================================
@st.cache_data
def load_database(filepath: str = "database.json") -> dict:
    """Memuat database klasifikasi batik dari file JSON"""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("❌ File `database.json` tidak ditemukan!")
        return {}
    except json.JSONDecodeError:
        st.error("❌ Format `database.json` tidak valid!")
        return {}

db_batik = load_database()

# =============================================================================
# LOAD MODEL (ResNet-18, CPU-Optimized)
# =============================================================================
@st.cache_resource
def load_model(
    model_path_primary: str = "batik_model_resnet_best.pth",
    model_path_fallback: str = "batik_model_best_V2_Unfitted.pth",
    num_classes: int = 20,
):
    """Memuat model ResNet-18 yang telah di-fine-tune"""
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
        )
        model_path = (
            model_path_primary
            if os.path.exists(model_path_primary)
            else model_path_fallback
        )
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

model = load_model()

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocessing gambar agar sesuai dengan input training ResNet-18"""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict(model: nn.Module, image_tensor: torch.Tensor, top_k: int = 3) -> list:
    """Menghasilkan top-K prediksi dengan confidence score"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        return [
            {
                "class_id": str(idx.item()),
                "confidence": prob.item() * 100,
                "raw_prob": prob.item(),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

# =============================================================================
# UI: IMAGE INPUT (Tabs)
# =============================================================================
tab_camera, tab_gallery = st.tabs(["📸 Kamera", "📁 Galeri"])
uploaded_file = None

with tab_camera:
    st.markdown("<p class='subtitle'>📸 Ambil Foto Langsung</p>", unsafe_allow_html=True)
    st.info("💡 Tips: Pastikan pencahayaan cukup dan motif batik terlihat jelas")
    camera_input = st.camera_input("Ambil Foto Batik", key="camera_input")
    if camera_input:
        uploaded_file = camera_input

with tab_gallery:
    st.markdown("<p class='subtitle'>📁 Unggah dari Galeri</p>", unsafe_allow_html=True)
    gallery_input = st.file_uploader(
        "Pilih gambar batik",
        type=["jpg", "jpeg", "png", "webp"],
        key="gallery_input",
        help="Format didukung: JPG, JPEG, PNG, WebP",
    )
    if gallery_input:
        uploaded_file = gallery_input

# =============================================================================
# MAIN PREDICTION LOGIC
# =============================================================================
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.container():
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="📷 Gambar yang dianalisis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Analisis Motif Batik", type="primary", use_container_width=True):
        with st.spinner("🤖 AI sedang menganalisis pola batik..."):
            try:
                img_tensor = preprocess_image(image)
                predictions = predict(model, img_tensor, top_k=3)

                st.divider()
                st.success("✅ Analisis Selesai!")

                for i, pred in enumerate(predictions):
                    class_id = pred["class_id"]
                    confidence = pred["confidence"]

                    if confidence >= 70:
                        tier, emoji = "high-confidence", "🎯"
                    elif confidence >= 40:
                        tier, emoji = "medium-confidence", "🤔"
                    else:
                        tier, emoji = "low-confidence", "❓"

                    if class_id in db_batik:
                        data = db_batik[class_id]
                        name = data.get("nama", f"Kelas {class_id}")
                        origin = data.get("asal", "Tidak diketahui")
                        philosophy = data.get("filosofi", "Tidak tersedia")
                    else:
                        name = f"Kelas {class_id} (Unlabeled)"
                        origin = "Data belum tersedia"
                        philosophy = "Silakan update database.json"

                    st.markdown(
                        f"""
                        <div class="prediction-card {tier}">
                            <h4>{emoji} #{i+1} - {name}</h4>
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">
                                <span class="confidence-label">Keyakinan</span>
                                <span class="confidence-value">{confidence:.1f}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width:{min(confidence, 100)}%"></div>
                            </div>
                            <p>
                                <strong>📍 Asal:</strong> {origin}<br>
                                <strong>📖 Filosofi:</strong> {philosophy}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if i == 0 and confidence >= 80:
                        st.balloons()

                if predictions[0]["confidence"] < 40:
                    st.warning(
                        "⚠️ Keyakinan rendah. Coba ambil foto dengan pencahayaan lebih baik atau sudut yang lebih jelas."
                    )

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
                st.info("💡 Pastikan model dan database.json sesuai dengan arsitektur ResNet-18")

# =============================================================================
# FOOTER & HELP SECTION
# =============================================================================
st.divider()

with st.expander("❓ Bantuan & Informasi"):
    st.markdown(
        """
        ### 🎯 Cara Menggunakan BatikLens
        1. **Ambil foto** motif batik dengan kamera atau unggah dari galeri
        2. **Pastikan**:
           - Motif batik terlihat jelas dan terpusat
           - Pencahayaan cukup (hindari bayangan berat)
           - Hindari foto blur atau terlalu jauh
        3. **Tekan "Analisis"** dan lihat 3 prediksi teratas

        ### 📊 Interpretasi Hasil
        - 🎯 **>70%**: Prediksi sangat akurat
        - 🤔 **40-70%**: Prediksi cukup meyakinkan
        - ❓ **<40%**: Coba ambil foto ulang dengan kondisi lebih baik

        ### 🔧 Teknis
        - **Model**: ResNet-18 Fine-Tuned
        - **Input**: 224×224 pixels, ImageNet normalization
        - **Akurasi Validasi**: ~74%
        - **Inference**: CPU-optimized untuk kompatibilitas luas
        """
    )

st.markdown(
    """
    <div style="text-align:center;color:var(--color-brown);opacity:0.7;font-size:0.8rem;margin-top:2rem">
        <p>🔬 BatikLens AI • Prototipe PKM-KC 2026</p>
        <p>Model: ResNet-18 • Accuracy: 73.98%</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# ERROR HANDLING
# =============================================================================
if model is None:
    st.error("⚠️ Model tidak dapat dimuat. Pastikan file `batik_model_resnet_best.pth` tersedia.")
    st.info("💡 Solusi: Jalankan training script terlebih dahulu untuk menghasilkan model.")

if not db_batik:
    st.warning("⚠️ Database kosong. Prediksi akan menampilkan ID kelas tanpa informasi detail.")
    st.info("💡 Solusi: Update file `database.json` dengan informasi motif batik.")