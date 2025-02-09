import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from symspellpy.symspellpy import SymSpell, Verbosity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump  # Import per salvare il modello
from sklearn.metrics import confusion_matrix

class CustomTweetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_edit_distance=2, prefix_length=7):
        """
        Inizializza il preprocessor definendo internamente i dizionari
        per abbreviazioni e emoji, e caricando SymSpell per la correzione ortografica.
        """
        # Dizionario delle abbreviazioni
        self.abbr_dict = {
            "u": "you", "r": "are", "pls": "please", "idk": "i do not know", "omg": "oh my god",
            "btw": "by the way", "gr8": "great", "lol": "laughing out loud", "brb": "be right back",
            "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing", "smh": "shaking my head",
            "fyi": "for your information", "tbh": "to be honest", "np": "no problem", "imo": "in my opinion",
            "imho": "in my humble opinion", "wtf": "what the fuck", "wth": "what the hell", "gg": "good game",
            "bff": "best friends forever", "ftw": "for the win", "afk": "away from keyboard", "nvm": "never mind",
            "ttyl": "talk to you later", "ikr": "i know, right?", "jk": "just kidding", "idc": "i do not care",
            "hmu": "hit me up", "wyd": "what you doing", "wbu": "what about you", "g2g": "got to go",
            "ty": "thank you", "yw": "you are welcome", "thx": "thanks", "yolo": "you only live once",
            "xoxo": "hugs and kisses", "bday": "birthday", "omw": "on my way", "pov": "point of view",
            "ily": "i love you", "ilu": "i love you", "dm": "direct message", "ggwp": "good game well played",
            "mfw": "my face when", "tfw": "that feeling when", "tldr": "too long did not read", "b4": "before",
            "bc": "because", "b/c": "because", "cuz": "because", "msg": "message", "fml": "fuck my life",
            "fb": "facebook", "ig": "instagram", "twt": "twitter", "faq": "frequently asked questions",
            "rip": "rest in peace", "irl": "in real life", "otp": "one true pairing", "hbd": "happy birthday",
            "grats": "congratulations", "ez": "easy", "glhf": "good luck have fun", "w/": "with",
            "w/o": "without", "bffl": "best friends for life", "bby": "baby", "bbygirl": "baby girl",
            "bbyboy": "baby boy", "tba": "to be announced", "tbc": "to be confirmed", "lmk": "let me know",
            "wdym": "what do you mean", "ttys": "talk to you soon", "tf": "the fuck", "wt": "what",
            "bcuz": "because", "wym": "what you mean", "fr": "for real", "rn": "right now",
            "prolly": "probably", "jfc": "jesus fucking christ", "stg": "swear to god", "ntm": "not too much",
            "smdh": "shaking my damn head", "lil": "little", "b": "be", "ttyn": "talk to you never",
            "dunno": "do not know", "goat": "greatest of all time", "tbf": "to be fair",
            "ffs": "for fuck's sake", "its": "it is"
        }

        # Dizionario delle emoji testuali
        self.emoji_dict = {
            ":)": "happy",
            ":-)": "happy",
            ":D": "joyful",
            ":-D": "joyful",
            ":(": "sad",
            ":-(": "sad",
            ";)": "wink",
            ";-)": "wink",
            ":P": "playful",
            ":-P": "playful",
            ":'(": "crying",
            ":-'(": "crying",
            "XD": "laughing",
            "xD": "laughing",
            ":-|": "neutral",
            ":|": "neutral"
        }

        # Parametri per SymSpell e caricamento del dizionario
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.symspell = SymSpell(self.max_edit_distance, self.prefix_length)
        dictionary_path = "../../data/frequency_dictionary_en_82_765.txt"  # Aggiorna il path se necessario
        try:
            self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        except Exception as e:
            print("Attenzione: impossibile caricare il dizionario SymSpell!", e)
            self.symspell = None

        # Altri elementi per il preprocessing
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Regex per rimozioni
        self.url_pattern = r'http\S+|www\S+'
        self.mention_pattern = r'@\w+'
        self.hashtag_symbol_pattern = r'#'
        self.special_chars_pattern = r'[^a-zA-Z\s]'  # Mantiene solo lettere e spazi
        self.wa_pattern = r'\bwa\b'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_texts = []
        for text in X:
            # 1. Lowercase
            text = text.lower()

            # 2. Rimozione URL
            text = re.sub(self.url_pattern, '', text)

            # 3. Rimozione menzioni (@username)
            text = re.sub(self.mention_pattern, '', text)

            # 4. Sostituzione entità HTML
            text = text.replace("&amp;", " ")
            text = text.replace("&quot;", " ")
            text = text.replace("&lt;", " ")
            text = text.replace("&gt;", " ")
            text = text.replace("&apos;", "'")

            # 5. Sostituzione emoji testuali
            for emoji, meaning in self.emoji_dict.items():
                text = re.sub(re.escape(emoji), meaning, text)

            # 6. Rimozione del simbolo '#' (mantiene la parola)
            text = re.sub(self.hashtag_symbol_pattern, '', text)

            # 7. Espansione delle abbreviazioni
            text = self.expand_abbreviations(text)

            # 8. Espansione delle contrazioni
            text = contractions.fix(text)

            # 9. Rimozione di caratteri speciali (mantieni solo lettere e spazi)
            text = re.sub(self.special_chars_pattern, ' ', text)

            # 10. Correzione ortografica tramite SymSpell
            text = self.correct_spelling(text)

            # 11. Normalizzazione degli spazi
            text = re.sub(r'\s+', ' ', text).strip()

            # 12. Tokenizzazione, Lemmatizzazione e rimozione delle stopwords
            tokens = word_tokenize(text)
            tokens_lemma = [self.lemmatizer.lemmatize(tok) for tok in tokens]
            tokens_clean = [t for t in tokens_lemma if t not in self.stop_words]

            # 13. Rimozione del token "wa"
            tokens_final = [t for t in tokens_clean if t != 'wa']

            # Ricostruzione del testo finale
            final_text = " ".join(tokens_final)
            cleaned_texts.append(final_text)
        return cleaned_texts

    def expand_abbreviations(self, text):
        tokens = text.split()
        expanded_tokens = [self.abbr_dict.get(tok, tok) for tok in tokens]
        return " ".join(expanded_tokens)

    def correct_spelling(self, text):
        tokens = text.split()
        corrected_tokens = []
        if self.symspell:
            for token in tokens:
                suggestions = self.symspell.lookup(token, Verbosity.CLOSEST, self.max_edit_distance)
                if suggestions:
                    corrected_tokens.append(suggestions[0].term)
                else:
                    corrected_tokens.append(token)
        else:
            corrected_tokens = tokens
        return " ".join(corrected_tokens)



# Creazione della Pipeline
pipeline = Pipeline([
    ("preproc", CustomTweetPreprocessor()),
    ("tfidf", TfidfVectorizer(min_df=5, max_df=0.9)),
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
# Esegue la cross validation calcolando accuracy, precision, recall
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro']
results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring_metrics, return_estimator=False)

acc_values = results['test_accuracy']
prec_values = results['test_precision_macro']
rec_values = results['test_recall_macro']

# Creazione di un array per identificare il numero di fold
folds = range(1, len(acc_values) + 1)



# Stampa dei valori per ogni fold
print("=== Risultati per fold ===")
for i in range(len(acc_values)):
    print(f"Fold {i+1}: Accuracy={acc_values[i]:.4f}, Precision={prec_values[i]:.4f}, Recall={rec_values[i]:.4f}")

print("\n=== Risultati medi e deviazione standard ===")
print(f"Media Accuracy: {np.mean(acc_values):.4f}  | Std: {np.std(acc_values):.4f}")
print(f"Media Precision: {np.mean(prec_values):.4f} | Std: {np.std(prec_values):.4f}")
print(f"Media Recall: {np.mean(rec_values):.4f}    | Std: {np.std(rec_values):.4f}")



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

