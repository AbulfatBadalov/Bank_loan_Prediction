
# Banka Kredisi Tahmin Uygulaması

Bu proje, bankacılık sektöründe kişilere kredi verilip verilmeyeceğini tahmin etmek amacıyla geliştirilmiş bir makine öğrenimi uygulamasıdır. Model, bir müşterinin demografik ve finansal özelliklerini kullanarak kredi alıp almayacağını tahmin eder.

## 🔍 Proje Hakkında

Kredi tahmin modeli, **Lojistik Regresyon** algoritması kullanılarak eğitilmiştir. Eğitim verisi olarak `Bank_Personal_Loan_Modelling.csv` dosyası kullanılmış ve dengesiz sınıflar için **SMOTE (Synthetic Minority Over-sampling Technique)** yöntemi uygulanmıştır.

## 📂 Proje Dosyaları

- `app.py`: Streamlit kullanılarak oluşturulmuş kullanıcı arayüzü.
- `bank_loan_log.py`: Veriyi hazırlayan, modeli eğiten ve kaydeden Python betiği.
- `Bank_Personal_Loan_Modelling (1).csv`: Modeli eğitmek için kullanılan veri seti.
- `logistic_model.pkl`: Eğitilen lojistik regresyon modelinin kaydedilmiş hali.
- `model_columns.pkl`: Modelin eğitildiği sütun bilgileri.
- `README.md`: Bu dökümantasyon dosyası.

## ⚙️ Kullanılan Kütüphaneler

- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- streamlit
- pickle

## 🚀 Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri kurun:

```bash
pip install pandas numpy scikit-learn imbalanced-learn streamlit
```

2. Modeli eğitmek için aşağıdaki komutu çalıştırın:

```bash
python bank_loan_log.py
```

3. Web arayüzünü başlatmak için:

```bash
streamlit run app.py
```

## 🧠 Model Girdileri

Aşağıdaki özelliklere göre tahmin yapılmaktadır:

- Age (yaş)
- Experience (iş tecrübesi)
- Income (gelir)
- Family (aile bireyi sayısı)
- CCAvg (ortalama kredi kartı harcaması)
- Education (1: Lisans, 2: Mezun, 3: Gelişmiş)
- Mortgage (ipotek)
- Securities Account (menkul kıymet hesabı)
- CD Account (vadeli mevduat hesabı)
- Online (online bankacılık kullanımı)
- CreditCard (kredi kartı kullanımı)

## ✅ Model Performansı

Model değerlendirmesi:
- **Accuracy**
- **Confusion Matrix**
- **ROC-AUC Skoru**
- **Precision / Recall / F1-Score**

## 🛠️ İleri Geliştirme Fikirleri

- Daha fazla algoritma ile model karşılaştırması yapılabilir (Random Forest, XGBoost, vb).
- Hyperparametre optimizasyonu yapılabilir.
- Modelin yeniden eğitilmesi için bir web paneli geliştirilebilir.
- Eksik veriler için daha gelişmiş imputasyon yöntemleri uygulanabilir.

## 👨‍💻 Geliştirici

Bu proje bir eğitim çalışması kapsamında oluşturulmuştur. Geliştirmeye ve katkıya açıktır.
