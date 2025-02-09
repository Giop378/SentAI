import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from symspellpy.symspellpy import SymSpell, Verbosity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump  # Import per salvare il modello
from src.preprocessor import CustomTweetPreprocessor
from sklearn.metrics import confusion_matrix

# Creazione della Pipeline
pipeline = Pipeline([
    ("preproc", CustomTweetPreprocessor()),
    ("bow", CountVectorizer(min_df=5, max_df=0.9)),
    ("clf", MultinomialNB())
])

# Caricamento del dataset
dataset = pd.read_csv("../../data/sentiment140_reduced_clean.csv")  # Aggiorna il path al file
dataset = dataset.dropna(subset=['text'])
dataset = dataset[dataset['text'].str.strip() != '']

X = dataset['text']
y = dataset['target']

# Definisce la cross validation con StratifiedKFold (k=10)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro']

# Esegue la cross validation calcolando accuracy, precision, recall
results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring_metrics, return_estimator=False)

acc_values = results['test_accuracy']
prec_values = results['test_precision_macro']
rec_values = results['test_recall_macro']

folds = range(1, len(acc_values) + 1)

# Stampa dei valori per ogni fold
print("=== Risultati per fold ===")
for i in range(len(acc_values)):
    print(f"Fold {i+1}: Accuracy={acc_values[i]:.4f}, Precision={prec_values[i]:.4f}, Recall={rec_values[i]:.4f}")

print("\n=== Risultati medi ===")
print(f"Media Accuracy: {np.mean(acc_values):.4f}")
print(f"Media Precision: {np.mean(prec_values):.4f}")
print(f"Media Recall: {np.mean(rec_values):.4f}")

# Generazione dei grafici a linee (Accuracy, Precision, Recall)
plt.figure(figsize=(6, 4))
plt.plot(folds, acc_values, marker='o', color='blue')
plt.title('Accuracy per fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(folds)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(folds, prec_values, marker='o', color='green')
plt.title('Precision per fold')
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.xticks(folds)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(folds, rec_values, marker='o', color='red')
plt.title('Recall per fold')
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.xticks(folds)
plt.grid(True)
plt.tight_layout()
plt.show()

# Creazione della tabella globale con medie (Accuracy, Precision, Recall)
global_accuracy = np.mean(acc_values)
global_precision = np.mean(prec_values)
global_recall = np.mean(rec_values)

fig_global, ax_global = plt.subplots(figsize=(4, 2))
ax_global.axis('off')
columns_global = ["Accuracy", "Precision", "Recall"]
table_data_global = [[
    f"{global_accuracy:.3f}",
    f"{global_precision:.3f}",
    f"{global_recall:.3f}"
]]
tbl_global = ax_global.table(cellText=table_data_global, colLabels=columns_global, loc='center')
tbl_global.auto_set_font_size(False)
tbl_global.set_fontsize(10)
plt.tight_layout()
plt.show()

# Addestramento finale sul dataset completo
pipeline.fit(X, y)

# Salvataggio del modello
model_filename = "naive_bayes_pipeline.joblib"
dump(pipeline, model_filename)
print(f"\nModello addestrato e salvato in '{model_filename}'.")


