import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# 1. SETUP TAMPILAN MOBILE-FRIENDLY
# Mengunci layout agar di tengah dan menyembunyikan sidebar otomatis di HP
st.set_page_config(page_title="BatikLens AI", page_icon="🔍", layout="centered", initial_sidebar_state="collapsed")

# Sedikit injeksi CSS agar jarak margin atas di HP tidak terlalu jauh (memaksimalkan layar)
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 BatikLens")
st.caption("Prototipe AI Identifikasi Batik - PKM-KC 2026")

# 2. LOAD DATABASE JSON
@st.cache_data
def load_database():
    with open('database.json', 'r') as file:
        return json.load(file)

db_batik = load_database()

# 3. LOAD MODEL AI (Di-cache agar tidak reload setiap ada interaksi)
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 20)
    # map_location='cpu' SANGAT PENTING agar bisa jalan di server gratisan yang tidak punya GPU
    model.load_state_dict(torch.load('batik_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 4. ATURAN TRANSLASI GAMBAR KE TENSOR
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# 5. UI INPUT: TAB UNTUK HP (Kamera vs Galeri)
# Tab jauh lebih ramah sentuhan (touch-friendly) di HP dibanding radio button
tab1, tab2 = st.tabs(["📸 Buka Kamera", "📁 Pilih dari Galeri"])

file_gambar = None

with tab1:
    st.write("Arahkan kamera HP Anda ke kain batik.")
    # Kamera langsung terbuka di browser HP
    input_kamera = st.camera_input("Ambil Foto Batik") 
    if input_kamera:
        file_gambar = input_kamera

with tab2:
    st.write("Atau unggah foto batik dari memori HP Anda.")
    # Membuka galeri foto HP
    input_galeri = st.file_uploader("Unggah Foto", type=["jpg", "jpeg", "png"])
    if input_galeri:
        file_gambar = input_galeri

# 6. LOGIKA PEMROSESAN AI
if file_gambar is not None:
    # Tampilkan gambar yang akan dianalisis menyesuaikan lebar layar HP
    image = Image.open(file_gambar).convert('RGB')
    st.image(image, caption="Gambar yang diproses", use_container_width=True)

    with st.spinner("AI sedang memindai pola matriks..."):
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            
            # Ubah output matriks mentah menjadi persentase (Softmax)
            probabilities = F.softmax(output[0], dim=0)
            confidence, pred_id = torch.max(probabilities, 0)
            
            id_tebakan = str(pred_id.item())
            persentase = confidence.item() * 100

    # 7. TAMPILKAN HASIL DARI DATABASE
    st.divider() # Garis pemisah UI
    
    if id_tebakan in db_batik:
        data = db_batik[id_tebakan]
        
        # Tampilkan hasil dengan format yang elegan
        st.success(f"✅ Analisis Selesai! (Tingkat Keyakinan: {persentase:.1f}%)")
        st.header(f"✨ {data['nama']}")
        st.markdown(f"**📍 Asal Daerah:** {data['asal']}")
        st.info(f"**📖 Filosofi:** {data['filosofi']}")
        
        # Tambahan efek balon perayaan jika AI sangat yakin (opsional, bagus untuk demo)
        if persentase > 80.0:
            st.balloons()
    else:
        st.warning(f"Terdeteksi sebagai ID {id_tebakan}, tapi data belum ada di database.json.")