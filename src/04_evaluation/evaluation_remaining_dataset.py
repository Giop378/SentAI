from src.preprocessor import CustomTweetPreprocessor  # Importa la classe necessaria
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Carica il modello salvato
model_filename = "../03_modeling/naive_bayes_pipeline.joblib"
pipeline = load(model_filename)

# Carica il dataset di test
test_dataset_path = "../../data/sentiment140_remaining_clean.csv"
dataset_test = pd.read_csv(test_dataset_path)
dataset_test = dataset_test.dropna(subset=['text', 'target'])
dataset_test = dataset_test[dataset_test['text'].str.strip() != '']
X_test = dataset_test['text']
y_test = dataset_test['target']

# Effettua le predizioni
y_pred = pipeline.predict(X_test)

# Calcolo delle metriche
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Stampa in console le metriche
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Visualizzazione delle metriche in tabella
fig_metrics, ax_metrics = plt.subplots(figsize=(4, 2))
ax_metrics.axis('off')
columns = ["Accuracy", "Precision", "Recall"]
table_data = [[f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}"]]
tbl_metrics = ax_metrics.table(cellText=table_data, colLabels=columns, loc='center')
tbl_metrics.auto_set_font_size(False)
tbl_metrics.set_fontsize(10)
plt.tight_layout()
plt.show()

# Calcolo e visualizzazione della matrice di confusione
cm = confusion_matrix(y_test, y_pred)
cm_list = cm.tolist()
flatten = [str(item) for row in cm_list for item in row]
fig_cm, ax_cm = plt.subplots(figsize=(8, 4))
ax_cm.axis('off')
columns_cm = ["Fold/Model", "TN", "FP", "FN", "TP"]
table_data_cm = [["Test Model"] + flatten]
tbl_cm = ax_cm.table(cellText=table_data_cm, colLabels=columns_cm, loc='center')
tbl_cm.auto_set_font_size(False)
tbl_cm.set_fontsize(8)
plt.tight_layout()
plt.show()
