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
    model.load_state_dict(torch.load('batik_model_best_V2_Unfitted.pth', map_location=torch.device('cpu')))
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
            
            # JURUS TOP-3 PREDICTIONS (Ambil 3 probabilitas tertinggi)
            top3_prob, top3_id = torch.topk(probabilities, 3)

    # 7. TAMPILKAN HASIL DARI DATABASE
    st.divider() # Garis pemisah UI
    st.success("✅ Analisis Selesai! Berikut 3 kemungkinan teratas:")
    
    # Looping untuk menampilkan 3 hasil prediksi
    for i in range(3):
        id_tebakan = str(top3_id[i].item())
        persentase = top3_prob[i].item() * 100
        
        # Cek apakah ID ada di database.json
        if id_tebakan in db_batik:
            data = db_batik[id_tebakan]
            
            # Membuat desain "Card" (Kartu) yang elegan untuk setiap tebakan
            with st.container():
                st.subheader(f"#{i+1} - {data['nama']}")
                
                # Menampilkan Bar Persentase Visual (Progress Bar)
                st.write(f"**Tingkat Keyakinan: {persentase:.1f}%**")
                # Streamlit progress bar butuh angka integer 0-100
                st.progress(int(persentase)) 
                
                st.markdown(f"**📍 Asal Daerah:** {data['asal']}")
                st.info(f"**📖 Filosofi:** {data['filosofi']}")
                st.write("---") # Garis pemisah antar kartu
                
                # Efek balon perayaan JIKA tebakan pertama sangat yakin (> 80%)
                if i == 0 and persentase > 80.0:
                    st.balloons()
        else:
            st.warning(f"Terdeteksi sebagai ID Kelas {id_tebakan} ({persentase:.1f}%), tapi data belum ada di database.json.")