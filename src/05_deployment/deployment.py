import tkinter as tk
from tkinter import messagebox
from joblib import load

MODEL_FILENAME = "../03_modeling/logistic_regression_tfidf_pipeline.joblib"
try:
    model = load(MODEL_FILENAME)
except Exception as e:
    messagebox.showerror("Errore", f"Impossibile caricare il modello:\n{e}")
    exit(1)

modern_font = ("Segoe UI", 12)
CONFIDENCE_THRESHOLD = 0.55


def evaluate_sentiment():
    text_input = text_field.get("1.0", "end-1c").strip()
    if len(text_input) > 140:
        messagebox.showerror("Errore", "La frase deve avere al massimo 140 caratteri!")
        return
    proba = model.predict_proba([text_input])[0]
    confidence = max(proba)
    prediction = model.predict([text_input])[0]

    if prediction == 1:
        message = "Sentiment positivo"
        color = "blue"
    else:
        message = "Sentiment negativo"
        color = "purple"

    if confidence < CONFIDENCE_THRESHOLD:
        message += f" (predizione incerta - Confidenza: {confidence * 100:.1f}%)"

    result_label.config(text=message, fg=color)


root = tk.Tk()
root.title("Analisi del Sentiment")
root.configure(bg="#E0F7FA")

instruction_label = tk.Label(root, text="Inserisci una frase o tweet (max 140 caratteri):",
                             bg="#E0F7FA", fg="#01579B", font=modern_font)
instruction_label.pack(padx=10, pady=10)

text_field = tk.Text(root, height=2, width=60, font=modern_font)
text_field.pack(padx=10, pady=5)

evaluate_button = tk.Button(root, text="Valuta Sentiment", command=evaluate_sentiment,
                            bg="#0288D1", fg="white", font=modern_font)
evaluate_button.pack(padx=10, pady=10)

result_label = tk.Label(root, text="", font=modern_font, bg="#E0F7FA")
result_label.pack(padx=10, pady=10)

root.mainloop()
