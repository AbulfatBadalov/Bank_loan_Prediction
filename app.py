import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Model ve sütunları yükle
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("Kredi Başvuru Tahmini")

# Kullanıcıdan veri al
age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
experience = st.number_input("İş Tecrübesi (yıl)", min_value=0, max_value=80, value=5)
income = st.number_input("Yıllık Gelir (bin $)", min_value=0, value=60)
ccavg = st.number_input("Kredi Kartı Harcaması (bin $)", min_value=0.0, value=1.5)
mortgage = st.number_input("Mortgage (bin $)", min_value=0.0, value=0.0)
securities_account = st.selectbox("Menkul Kıymet Hesabı", [0, 1])
cd_account = st.selectbox("Vadeli Mevduat Hesabı", [0, 1])
online = st.selectbox("Online Kullanıcı", [0, 1])
creditcard = st.selectbox("Kredi Kartı", [0, 1])

# Kategorik veriler
education = st.selectbox("Eğitim Durumu", ["Undergrad", "Graduate", "Advanced/Professional"])
family = st.selectbox("Aile Üyesi Sayısı", [1, 2, 3, 4])

if st.button("Tahmin Et"):
    # Kullanıcı girdilerini DataFrame'e çevir
    input_data = {
        "Age": age,
        "Experience": experience,
        "Income": income,
        "CCAvg": ccavg,
        "Mortgage": mortgage,
        "Securities Account": securities_account,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": creditcard,
        "Education_Graduate": 1 if education == "Graduate" else 0,
        "Education_Advanced/Professional": 1 if education == "Advanced/Professional" else 0,
        "Family_2": 1 if family == 2 else 0,
        "Family_3": 1 if family == 3 else 0,
        "Family_4": 1 if family == 4 else 0,
    }

    input_df = pd.DataFrame([input_data])

    # Eksik kolonları sıfırla
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Tahmin yap
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Kredi verilebilir (olasılık: {probability:.2f})")
    else:
        st.error(f"❌ Kredi verilmez (olasılık: {probability:.2f})")
