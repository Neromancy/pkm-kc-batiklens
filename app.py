import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import os

# =====================================================================
# 1. PAGE CONFIGURATION (Mobile-Optimized)
# =====================================================================
st.set_page_config(
    page_title="BatikLens AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# BatikLens AI 🔍\nPrototipe PKM-KC 2026\n\nModel: ResNet-18 Fine-Tuned"
    }
)

# =====================================================================
# 2. CUSTOM CSS (Using Provided Color Palette)
# =====================================================================
# =====================================================================
# 2. CUSTOM CSS (Dark Brown Text + Color Palette)
# =====================================================================
st.markdown("""
    <style>
        /* Color Variables */
        :root {
            --color-dark: #3E333F;
            --color-brown: #5D4E60;
            --color-red: #FC4442;
            --color-yellow: #F0E19E;
            --color-light: #F2F2F2;
        }
        
        /* Global Background */
        .stApp { 
            background-color: var(--color-light);
        }
        
        /* MAIN TEXT - GELAP COKLAT */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: var(--color-dark) !important;
        }
        
        .main .block-container { 
            padding-top: 1rem; 
            padding-bottom: 2rem;
            max-width: 600px;
        }
        
        /* Title Styling - GELAP COKLAT PEKAT */
        h1 { 
            color: var(--color-dark) !important; 
            text-align: center; 
            font-size: 2rem !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: none;
        }
        
        .subtitle { 
            color: var(--color-brown) !important; 
            text-align: center; 
            margin-bottom: 1.5rem;
            font-size: 1rem;
            font-weight: 500;
        }
        
        /* Prediction Cards */
        .prediction-card {
            background: white;
            border-radius: 16px;
            padding: 1.25rem;
            margin: 0.75rem 0;
            box-shadow: 0 4px 12px rgba(62, 51, 63, 0.15);
            border-left: 5px solid var(--color-dark);
        }
        
        .prediction-card.high-confidence { 
            border-left-color: var(--color-red); 
        }
        
        .prediction-card.medium-confidence { 
            border-left-color: var(--color-yellow); 
        }
        
        .prediction-card h4 {
            color: var(--color-dark) !important;
            margin: 0 0 0.75rem 0;
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        /* Confidence Text - GELAP */
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
        
        /* Confidence Bar */
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
        
        .prediction-card strong {
            color: var(--color-dark) !important;
        }
        
        /* Image Container */
        .image-container {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(62, 51, 63, 0.15);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--color-dark);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
        }
        
        .stButton>button:hover {
            background-color: var(--color-red);
            color: white !important;
        }
        
        /* Tabs - GELAP COKLAT */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 0.5rem; 
            background: white;
            border-radius: 12px;
            padding: 0.25rem;
            box-shadow: 0 2px 8px rgba(62, 51, 63, 0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: var(--color-light);
            border-radius: 10px;
            color: var(--color-dark) !important;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--color-dark);
            color: white !important;
        }
        
        /* Info/Alert Boxes - TEKS GELAP */
        .stAlert {
            border-radius: 12px;
            border: none;
            color: var(--color-dark) !important;
        }
        
        .stAlert * {
            color: var(--color-dark) !important;
        }
        
        .stInfo, .stInfo * {
            color: var(--color-dark) !important;
            background-color: rgba(240, 225, 158, 0.3) !important;
        }
        
        .stSuccess, .stSuccess * {
            color: var(--color-dark) !important;
            background-color: rgba(242, 242, 242, 0.9) !important;
        }
        
        .stWarning, .stWarning * {
            color: var(--color-dark) !important;
        }
        
        /* File Uploader - TEKS GELAP */
        .stFileUploader {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(62, 51, 63, 0.1);
        }
        
        .stFileUploader * {
            color: var(--color-dark) !important;
        }
        
        /* Expander - TEKS GELAP */
        .streamlit-expanderHeader, .streamlit-expanderHeader * {
            color: var(--color-dark) !important;
            background: white;
        }
        
        .streamlit-expanderContent, .streamlit-expanderContent * {
            color: var(--color-brown) !important;
            background: white;
        }
        
        /* Footer - GELAP COKLAT */
        footer {
            color: var(--color-brown) !important;
        }
        
        footer * {
            color: var(--color-brown) !important;
        }
        
        /* Mobile Optimization */
        @media (max-width: 768px) {
            h1 { font-size: 1.5rem !important; }
            .subtitle { font-size: 0.9rem; }
            .prediction-card { padding: 1rem; }
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. HEADER SECTION
# =====================================================================
st.markdown("<h1>🔍 BatikLens AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Identifikasi Motif Batik dengan Deep Learning • ResNet-18</p>", unsafe_allow_html=True)

# =====================================================================
# 4. LOAD DATABASE (Cached)
# =====================================================================
@st.cache_data
def load_database():
    """Load batik classification database from JSON"""
    try:
        with open('database.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("❌ File `database.json` tidak ditemukan!")
        return {}
    except json.JSONDecodeError:
        st.error("❌ Format `database.json` tidak valid!")
        return {}

db_batik = load_database()

# =====================================================================
# 5. LOAD MODEL (ResNet-18, CPU-Optimized)
# =====================================================================
@st.cache_resource
def load_model():
    """Load fine-tuned ResNet-18 model with CPU compatibility"""
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 20)
        )
        
        model_path = 'batik_model_resnet_best.pth'
        if not os.path.exists(model_path):
            model_path = 'batik_model_best_V2_Unfitted.pth'
            
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

model = load_model()

# =====================================================================
# 6. IMAGE PREPROCESSING
# =====================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image to match ResNet-18 training transforms"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# =====================================================================
# 7. PREDICTION FUNCTION
# =====================================================================
def predict(model, image_tensor, top_k=3):
    """Generate top-K predictions with confidence scores"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                'class_id': str(idx.item()),
                'confidence': prob.item() * 100,
                'raw_prob': prob.item()
            })
        return results

# =====================================================================
# 8. UI: IMAGE INPUT
# =====================================================================
tab1, tab2 = st.tabs(["📸 Kamera", "📁 Galeri"])

uploaded_file = None

with tab1:
    st.markdown("### 📸 Ambil Foto Langsung")
    st.info("💡 Tips: Pastikan pencahayaan cukup dan motif batik terlihat jelas")
    camera_file = st.camera_input("Ambil Foto Batik", key="camera_input")
    if camera_file:
        uploaded_file = camera_file

with tab2:
    st.markdown("### 📁 Unggah dari Galeri")
    gallery_file = st.file_uploader(
        "Pilih gambar batik",
        type=["jpg", "jpeg", "png", "webp"],
        key="gallery_input",
        help="Format didukung: JPG, JPEG, PNG, WebP"
    )
    if gallery_file:
        uploaded_file = gallery_file

# =====================================================================
# 9. MAIN PREDICTION LOGIC
# =====================================================================
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    with st.container():
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="📷 Gambar yang dianalisis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Analisis Motif Batik", type="primary", use_container_width=True):
        with st.spinner("🤖 AI sedang menganalisis pola batik..."):
            try:
                img_tensor = preprocess_image(image)
                predictions = predict(model, img_tensor, top_k=3)
                
                st.divider()
                st.success("✅ Analisis Selesai!")
                
                for i, pred in enumerate(predictions):
                    class_id = pred['class_id']
                    confidence = pred['confidence']
                    
                    if confidence >= 70:
                        tier = "high-confidence"
                        emoji = "🎯"
                    elif confidence >= 40:
                        tier = "medium-confidence"
                        emoji = "🤔"
                    else:
                        tier = "low-confidence"
                        emoji = "❓"
                    
                    if class_id in db_batik:
                        data = db_batik[class_id]
                        name = data.get('nama', f'Kelas {class_id}')
                        origin = data.get('asal', 'Tidak diketahui')
                        philosophy = data.get('filosofi', 'Tidak tersedia')
                    else:
                        name = f'Kelas {class_id} (Unlabeled)'
                        origin = 'Data belum tersedia'
                        philosophy = 'Silakan update database.json'
                    
                    st.markdown(f"""
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
                    """, unsafe_allow_html=True)
                    
                    if i == 0 and confidence >= 80:
                        st.balloons()
                
                if predictions[0]['confidence'] < 40:
                    st.warning("⚠️ Keyakinan rendah. Coba ambil foto dengan pencahayaan lebih baik atau sudut yang lebih jelas.")
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
                st.info("💡 Pastikan model dan database.json sesuai dengan arsitektur ResNet-18")

# =====================================================================
# 10. FOOTER & HELP SECTION
# =====================================================================
st.divider()

with st.expander("❓ Bantuan & Informasi"):
    st.markdown("""
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
    """)

st.markdown("""
    <div style="text-align:center;color:#3E333F;opacity:0.7;font-size:0.8rem;margin-top:2rem">
        <p>🔬 BatikLens AI • Prototipe PKM-KC 2026</p>
        <p>Model: ResNet-18 • Accuracy: 73.98%</p>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# 11. ERROR HANDLING
# =====================================================================
if model is None:
    st.error("⚠️ Model tidak dapat dimuat. Pastikan file `batik_model_resnet_best.pth` tersedia.")
    st.info("💡 Solusi: Jalankan training script terlebih dahulu untuk menghasilkan model.")

if not db_batik:
    st.warning("⚠️ Database kosong. Prediksi akan menampilkan ID kelas tanpa informasi detail.")
    st.info("💡 Solusi: Update file `database.json` dengan informasi motif batik.")