# 📊 Customer Churn Dashboard

Dieses Projekt ist ein **interaktives Dashboard**, mit dem sich vorhersagen lässt, ob ein Kunde wahrscheinlich kündigen wird (Churn).  
Das Ganze basiert auf dem bekannten **Telco Customer Churn Datensatz** und einem trainierten Machine-Learning-Modell (Random Forest).  

Die Idee dahinter: Statt ein Modell nur im Jupyter Notebook laufen zu lassen, kann man es hier in einer einfachen Oberfläche ausprobieren – einzeln oder mit ganzen CSV-Dateien.

---

## 🔧 Funktionen

- **Einzelkunde**: Eingabe von Vertragsdauer, Kosten, Vertragstyp usw. → das Dashboard gibt sofort die Churn-Wahrscheinlichkeit zurück.  
- **Batch-Upload**: CSV-Datei hochladen → Vorhersage für viele Kunden gleichzeitig, inklusive Download der Ergebnisse.  
- **Model Insights**: Anzeige der wichtigsten Einflussfaktoren des Modells (Feature Importances).  
- **Infos**: Kurze Übersicht zum Projekt.  

---

## 🛠 Verwendete Technologien

- Python  
- [Streamlit](https://streamlit.io/) für das Dashboard  
- [scikit-learn](https://scikit-learn.org/) für das Modell  
- pandas für Datenaufbereitung  
- plotly für Visualisierungen  
- joblib zum Speichern und Laden des Modells  

---

## 📂 Projektaufbau

```
.
├── app.py                  # Streamlit-App (Dashboard)
├── requirements.txt        # Python-Abhängigkeiten
├── README.md               # Projektdokumentation
│
├── data/                   # Beispieldaten (CSV)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/                 # Trainiertes Modell
│   ├── model.pkl
│   └── model_features.pkl
│
└── src/
    └── Modell vorbereiten.py   # Skript zum Trainieren und Speichern des Modells
```

---

## 🚀 Installation und Start

1. **Repository klonen**  
   ```bash
   git clone https://github.com/<DEIN_USERNAME>/kunden-churn-dashboard-prototyp.git
   cd kunden-churn-dashboard-prototyp
   ```

2. **Virtuelle Umgebung anlegen** (empfohlen)  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Abhängigkeiten installieren**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Dashboard starten**  
   ```bash
   streamlit run app.py
   ```

   Danach ist das Dashboard im Browser erreichbar unter  
   👉 `http://localhost:8501`

---

## 📊 Modell Training

Das Skript `src/Modell vorbereiten.py` führt folgende Schritte aus:  
- Daten laden und bereinigen  
- Kategorische Variablen in Dummy-Variablen umwandeln  
- Random Forest trainieren  
- Modell und Feature-Liste als `.pkl` speichern  

So können die gleichen Features später im Dashboard verwendet werden.

---

## ℹ️ Hinweise

- In `.gitignore` sind Ordner wie `.venv/`, `.idea/` und Cache-Dateien ausgeschlossen, damit das Repository sauber bleibt.  
- Der Datensatz im Ordner `data/` ist nur ein Beispiel. Eigene CSV-Dateien können im Dashboard hochgeladen werden.  
- Das Projekt ist so aufgebaut, dass es leicht auf **Streamlit Cloud** oder einem eigenen Server deployt werden kann.  

---

## 👤 Autor

Erstellt von mir als Lern- und Praxisprojekt im Bereich **Data Science / Machine Learning**.  
Das Dashboard soll zeigen, wie man ein trainiertes Modell in einer benutzerfreundlichen Oberfläche einsetzen kann.  
