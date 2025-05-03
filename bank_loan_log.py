# Gerekli kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

# 1. Veriyi oku
data = pd.read_csv("Bank_Personal_Loan_Modelling (1).csv")

# 2. Gereksiz sütunları at
data.drop(columns=[col for col in ['ID', 'ZIP Code'] if col in data.columns], inplace=True)

# 3. Kategorik sütunları one-hot encode et (drop_first=False, çünkü binary değil)
data = pd.get_dummies(data, columns=['Education', 'Family'], drop_first=False)

# 4. Özellik ve hedef
X = data.drop('Personal Loan', axis=1)
y = data['Personal Loan']

# 5. Eğitim-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. SMOTE ile dengesizlik düzeltme
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 7. Lojistik regresyon modeli eğitimi
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_resampled, y_train_resampled)

# 8. Tahminler
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

# 9. Değerlendirme
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# 10. Modeli ve sütun adlarını kaydet
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
