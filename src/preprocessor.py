# src/preprocessors.py
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from symspellpy.symspellpy import SymSpell, Verbosity

class CustomTweetPreprocessor:
    def __init__(self, max_edit_distance=2, prefix_length=7):
        # Dizionario delle abbreviazioni
        self.abbr_dict = {
            "u": "you", "r": "are", "pls": "please", "idk": "i do not know", "omg": "oh my god",
            # ... (continua con le abbreviazioni)
        }
        # Dizionario delle emoji testuali
        self.emoji_dict = {
            ":)": "happy",
            ":-)": "happy",
            ":D": "joyful",
            ":-D": "joyful",
            ":(": "sad",
            ":-(": "sad",
            # ... (continua con le emoji)
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
            text = text.replace("&amp;", " ").replace("&quot;", " ").replace("&lt;", " ").replace("&gt;", " ").replace("&apos;", "'")

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
