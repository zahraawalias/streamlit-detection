import streamlit as st
import numpy as np
import pickle
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====== Load Resources ======
@st.cache_resource
def load_artifacts():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    le_binary = joblib.load("binary_label_encoder.pkl")
    le_multi = joblib.load("multiclass_label_encoder.pkl")
    model_binary = load_model("model_binary.h5")
    model_multi = load_model("model_multiclass.h5")
    return tokenizer, le_binary, le_multi, model_binary, model_multi

tokenizer, le_binary, le_multi, model_binary, model_multi = load_artifacts()
max_len = 256

# ====== Preprocessing ======
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ====== Prediction Function ======
def predict_text(text):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    # prediksi binary: model memberi probability untuk label index=1
    pred_binary_prob = float(model_binary.predict(padded)[0][0])

    # mapping label encoder (aman kalau mapping kebalik)
    label_idx0 = le_binary.inverse_transform([0])[0]
    label_idx1 = le_binary.inverse_transform([1])[0]

    # buat mapping label -> probabilitas
    prob_by_label = {
        label_idx1: pred_binary_prob,        # prob untuk index 1
        label_idx0: 1.0 - pred_binary_prob   # prob untuk index 0
    }

    # tentukan label hasil (threshold 0.5 pada prob index=1)
    pred_index = 1 if pred_binary_prob > 0.5 else 0
    pred_binary_label = le_binary.inverse_transform([pred_index])[0]

    # multiclass hanya jika predicted abusive (cek nama label)
    if pred_binary_label.lower().startswith("abusive"):
        pred_multi_prob = model_multi.predict(padded)[0]
        pred_multi_idx = int(np.argmax(pred_multi_prob))
        pred_multi_label = le_multi.inverse_transform([pred_multi_idx])[0]
    else:
        pred_multi_label, pred_multi_prob = None, None

    return {
        "pred_binary_label": pred_binary_label,
        "pred_binary_prob": pred_binary_prob,
        "prob_by_label": prob_by_label,
        "pred_multi_label": pred_multi_label,
        "pred_multi_prob": pred_multi_prob.tolist() if pred_multi_prob is not None else None,
    }

# ====== Streamlit UI ======
st.title("Deteksi Ujaran Kekerasan dan Klasifikasi Multikelas")
st.caption("Versi build: 00b745a — revisi sblm cetak")

st.markdown("""
Masukkan teks untuk dideteksi apakah mengandung ujaran kekerasan (abusive) atau tidak.  
Jika **abusive**, sistem juga akan mengklasifikasikan jenis ujaran kekerasannya (fisik, ekonomi, psikologis, atau seksual).
""")

user_input = st.text_area("Masukkan teks di sini:", height=150)

if st.button("Deteksi"):
    if not user_input.strip():
        st.warning("Teks input tidak boleh kosong!")
    else:
        with st.spinner("Memproses..."):
            results = predict_text(user_input)

        # Ambil mapping probabilitas yang benar
        prob_by_label = results["prob_by_label"]

        # Susun urutan tampilan biner: prefer 'not_abusive' dulu jika ada
        classes_bin = list(le_binary.classes_)
        display_order = []
        if "not_abusive" in classes_bin:
            display_order.append("not_abusive")
        if "abusive" in classes_bin:
            display_order.append("abusive")
        # tambahkan label lain bila ada
        for c in classes_bin:
            if c not in display_order:
                display_order.append(c)

        # Tampilkan hasil akhir (tanpa meng-echo teks input)
        pred_binary = results["pred_binary_label"].replace("_", " ").title()
        output_lines = []
        if results["pred_binary_label"].lower().startswith("abusive"):
            pred_multi = results["pred_multi_label"].replace("_", " ").title()
            output_lines.append(f"**Hasil Deteksi Akhir:** {pred_binary} ({pred_multi})")
        else:
            output_lines.append(f"**Hasil Deteksi Akhir:** {pred_binary}")

        output_lines.append("")  # baris kosong
        output_lines.append("**Probabilitas Klasifikasi Biner:**")
        for cls in display_order:
            pretty = cls.replace("_", " ").title()
            output_lines.append(f"- {pretty}: {prob_by_label[cls]:.4f}")

        # Jika abusive, tambahkan probabilitas multikelas
        if results["pred_multi_prob"] is not None:
            output_lines.append("") 
            output_lines.append("**Probabilitas Klasifikasi Multikelas:**")
            for idx, cls in enumerate(le_multi.classes_):
                pretty = cls.replace("_", " ").title()
                output_lines.append(f"- {pretty}: {results['pred_multi_prob'][idx]:.4f}")

        st.markdown("\n".join(output_lines))
