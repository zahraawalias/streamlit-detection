import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# ====== Load Resources ======
@st.cache_resource
def load_artifacts():
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    # Load label encoders
    le_binary = joblib.load("binary_label_encoder.pkl")
    le_multi = joblib.load("multiclass_label_encoder.pkl")
    # Load models
    model_binary = load_model("model_binary.h5")
    model_multi = load_model("model_multiclass.h5")
    return tokenizer, le_binary, le_multi, model_binary, model_multi

tokenizer, le_binary, le_multi, model_binary, model_multi = load_artifacts()

max_len = 256

# ====== Preprocessing text function ======
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove urls
    text = re.sub(r'\@\w+|\#','', text)  # remove @mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ====== Predict function ======
def predict_text(text):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Prediksi Binary
    pred_binary_prob = float(model_binary.predict(padded)[0][0])  # nilai asli
    pred_binary_label = le_binary.inverse_transform([1 if pred_binary_prob > 0.5 else 0])[0]

    if pred_binary_label == 'not_abusive':
        pred_multi_label, pred_multi_prob = None, None
    else:
        pred_multi_prob = model_multi.predict(padded)[0]
        pred_multi_idx = int(np.argmax(pred_multi_prob))
        pred_multi_label = le_multi.inverse_transform([pred_multi_idx])[0]

    return {
        "clean_text": clean_text,
        "pred_binary_label": pred_binary_label,
        "pred_binary_prob": pred_binary_prob,
        "pred_multi_label": pred_multi_label,
        "pred_multi_prob": pred_multi_prob.tolist() if pred_multi_prob is not None else None
    }


# ====== Streamlit UI ======
st.title("Deteksi Ujaran Kekerasan dan Klasifikasi Multiclass")

st.markdown("""
Masukkan teks untuk dideteksi apakah mengandung ujaran kekerasan (abusive) atau tidak (not abusive).  
Jika abusive, sistem juga akan mengklasifikasikan jenis ujaran kekerasannya (fisik, ekonomi, psikologis, seksual).
""")

user_input = st.text_area("Masukkan teks di sini:", height=150)
if st.button("Deteksi"):
    if not user_input.strip():
        st.warning("Teks input tidak boleh kosong!")
    else:
        with st.spinner("Memproses..."):
            results = predict_text(user_input)

        st.subheader("Hasil Deteksi")
        st.write(f"**Teks setelah preprocessing:** {results['clean_text']}")
        st.write(f"**Label Binary:** {results['pred_binary_label'].upper()}")
        st.write(f"**Probabilitas Binary:** {results['pred_binary_prob']:.4f}")

        if results['pred_multi_label']:
            st.write(f"**Label Multiclass:** {results['pred_multi_label'].upper()}")
            st.write("**Probabilitas tiap kelas:**")
            prob_df = pd.DataFrame({
                "Kelas": le_multi.classes_,
                "Probabilitas": [round(p, 4) for p in results['pred_multi_prob']]
            })
            st.table(prob_df)
        else:
            st.info("Teks ini diprediksi NOT ABUSIVE, jadi tidak ada klasifikasi multikelas.")
