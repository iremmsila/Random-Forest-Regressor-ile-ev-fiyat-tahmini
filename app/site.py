import streamlit as st
import pandas as pd
import numpy as np
import pickle  # model dosyasını yüklemek için

from ev import label_encoder

# Streamlit sayfa ayarları
st.set_page_config(page_title="Konut Fiyatları Tahmini", page_icon=":house:")  # sayfas başlığı ve simgesi

# Session state(oturum durumu)  veri saklama ve bu verileri istenildiğinde çağırma amaçlı kullanılan yapıdır.

def initialize_session_states():
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

@st.cache_resource
def load_model(filepath: str):
    with st.spinner('Model yükleniyor...'):
        return pickle.load(open(filepath, 'rb'))

# Model yükleme
loaded_model = load_model('random_forest_model.pkl')

# Session state(oturum) başlatma
initialize_session_states()

# Arka plan fotoğrafı
def add_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


image_url = "https://images.pexels.com/photos/358636/pexels-photo-358636.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
add_bg_image(image_url)

# Sayfa düzeni ve giriş verileri
st.title("Kaliforniya Konut Fiyatları Tahmini")
st.markdown("""
##### Kaliforniya Konut Fiyatlarını Tahmin Etme Web Uygulaması

Bu uygulama, makine öğrenimini kullanarak evin fiyatını tahmin eder. 
Önceden eğitilmiş bir Rastgele Orman modelini yükler ve bu model, 
bir evin fiyatını tahmin etmek için oda sayısı, yatak odası sayısı, 
evin bulunduğu mahallenin nüfusu ve en yakın şehre uzaklık gibi çeşitli özellikleri giriş olarak alır.
""")

# Kullanıcıdan girdileri alma
st.header("Konutun özelliklerini girin:")
longitude = st.number_input("Boylam", value=-120.0, step=0.1)
latitude = st.number_input("Enlem", value=35.0, step=0.1)
housing_median_age = st.number_input("Konut Medyan Yaşı", value=30, step=1)
total_rooms = st.number_input("Bloktaki Toplam Oda Sayısı", value=1000, step=1)
total_bedrooms = st.number_input("Bloktaki Toplam Yatak Odası Sayısı", value=200, step=1)
population = st.number_input("Mahallenin Nüfusu", value=500, step=1)
households = st.number_input("Hane Sayısı", value=150, step=1)
median_income = st.number_input("Medyan Gelir (on binlerce USD)", value=5.0, step=0.1)
ocean_proximity = st.selectbox("Okyanusa Yakınlık", ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Tahmin butonu
button = st.button("Tahmin Et")

# Tahmin işlemi
if button:
    if total_bedrooms > total_rooms:
        st.error('Hata: Toplam yatak odası sayısı toplam oda sayısından fazla olamaz.')
    else:
        input_data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": label_encoder.transform([ocean_proximity])[0],
        }
        print(ocean_proximity)

        input_df = pd.DataFrame([input_data])

        prediction = loaded_model.predict(input_df).squeeze()
        st.session_state['prediction'] = prediction
        st.success("Tahmin tamamlandı!")

# Tahmin sonucunu gösterme
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 34px;
        color: green;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.metric(label='Medyan Konut Değeri', value=f"$ {pred:.2f}")
    st.metric(label="kod",value=f"${label_encoder.transform([ocean_proximity])[0]}")
