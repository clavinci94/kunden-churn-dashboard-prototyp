# ========================
# 1. Libraries laden
# ========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ========================
# 2. Daten laden
# ========================
data = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ========================
# 3. Daten vorbereiten
# ========================
print("🧹 Datenvorbereitung...")

# 'TotalCharges' in numerisch umwandeln
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# CustomerID entfernen (bringt nichts für Vorhersage)
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Zielvariable Churn in 0/1 umwandeln
y = data['Churn'].map({'Yes': 1, 'No': 0})

# Kategorische Features in Zahlen encoden
X = data.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)   # One-Hot-Encoding

# ========================
# 4. Train/Test Split
# ========================
print("📊 Split in Train/Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 5. Modell trainieren
# ========================
print("🤖 Trainiere RandomForest Modell...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# ========================
# 6. Modell evaluieren
# ========================
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

# ========================
# 7. Modell + Features speichern
# ========================
print("💾 Speichere Modell und Features...")

# sicherstellen, dass Models-Ordner existiert
os.makedirs("models", exist_ok=True)

# Modell speichern
joblib.dump(rf_model, "models/model.pkl")

# Feature-Namen speichern (für Konsistenz in app.py)
joblib.dump(list(X_train.columns), "models/model_features.pkl")

print("🎉 Modell und Features erfolgreich gespeichert!")
