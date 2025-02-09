#%%
#import delle librerie necessarie
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import contractions
from symspellpy import SymSpell, Verbosity
nltk.download('punkt_tab')
import wordcloud
from wordcloud import WordCloud

#%% md
# # Caricamento e visualizzazione del dataset
#%%
#Caricamento e visualizzazione del dataset
dataset = pd.read_csv("../../data/sentiment140_reduced.csv", encoding="ISO-8859-1")
dataset
#%% md
# # Rimozione colonne non necessarie e conversione del target da 4 a 1 per i tweet positivi
#%%
#Modifica del dataset_reduced per la rimozione delle colonne non necessarie
dataset = dataset[['text', 'target']]
dataset = dataset.copy()
dataset
#%%
#Sostituzione dei valori della colonna 'target': 0 rimane 0 (negativo) e 4 diventa 1 (positivo)
dataset['target'] = dataset['target'].apply(lambda x: 1 if x == 4 else 0)
dataset
#Stampa distribuzione delle classi
print("Distribuzione delle classi:")
print(dataset['target'].value_counts())

#%% md
# # Rimozione duplicati
#%%
#Rimozione duplicati individuati durante la fase di data understanding (vengono rimossi i tweet che hanno lo stesso testo di altri)
# Conta il numero totale di duplicati nella colonna 'text'
numero_duplicati = dataset['text'].duplicated().sum()
print("Numero totale di duplicati nella colonna 'text':", numero_duplicati)

# Rimuovi i duplicati
dataset.drop_duplicates(subset='text', inplace=True)

# Conta il numero totale di duplicati nella colonna 'text' dopo la rimozione
numero_duplicati = dataset['text'].duplicated().sum()
print("Numero totale di duplicati nella colonna 'text' dopo la rimozione:", numero_duplicati)
#%%
#Verifica dell'eliminazione dei 296 duplicati
dataset.shape
#%% md
# # Filtraggio tweet con lunghezza anomala
#%%
#Visualizza il numero di tweet con lunghezza maggiore di 140 caratteri
print("Numero di tweet con lunghezza maggiore di 140 caratteri:", dataset[dataset['text'].str.len() > 140].shape[0])

#Rimozione dei tweet con lunghezza maggiore di 140 caratteri
dataset = dataset[dataset['text'].str.len() <= 140]

dataset.shape

#Numero di tweet con lunghezza maggiore di 140 caratteri dopo la rimozione
print("Numero di tweet con lunghezza maggiore di 140 caratteri dopo la rimozione:", dataset[dataset['text'].str.len() > 140].shape[0])
#%%
#Visualizza il numero di tweet con lunghezza minore di 10 caratteri
print("Numero di tweet con lunghezza minore di 10 caratteri:", dataset[dataset['text'].str.len() < 10].shape[0])

#Stampa tutti i tweet con lunghezza minore di 10 caratteri
print("Tweet con lunghezza minore di 10 caratteri:")
print(dataset[dataset['text'].str.len() < 10]['text'].tolist())

#Rimozione dei tweet con meno di 10 caratteri e che contengono una menzione
def contains_mention(text):
    return bool(re.search(r"@\w+", text))
short_with_mention = dataset[(dataset['text'].str.len() < 10) & (dataset['text'].apply(contains_mention))]

print("Numero di tweet con meno di 10 caratteri e che contengono una menzione:", short_with_mention.shape[0])

dataset = dataset[~((dataset['text'].str.len() < 10) & (dataset['text'].apply(contains_mention)))]

# Numero di tweet con lunghezza minore di 10 caratteri dopo la rimozione
print("Numero di tweet con lunghezza minore di 10 caratteri dopo la rimozione:", dataset[dataset['text'].str.len() < 10].shape[0])
#%%
dataset.shape #Numero di tweet rimanenti
#%%
#Visualizzazione del dataset dopo la pulizia
dataset
#%%
#Salvataggio del dataset su cui sarà fatto il training (deve essere fatta ancora la pulizia dei tweet)
dataset.to_csv("../../data/sentiment140_reduced_clean.csv", index=False)
#%%
#Caricamento del dataset pulito
dataset = pd.read_csv("../../data/sentiment140_reduced_clean.csv")
dataset
#%%
sentiment_counts = dataset['target'].value_counts()

# Stampa del numero di tweet per ciascuna classe
print("\nNumero di tweet per ciascuna classe:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")

# Creazione del diagramma a barre con posizioni corrette
plt.figure(figsize=(8, 5))

bar_positions = range(len(sentiment_counts))
plt.bar(bar_positions, sentiment_counts.values, color='skyblue', edgecolor='black')

# Imposta le etichette personalizzate per l'asse X
plt.xticks(bar_positions, ["Negativo (=0)", "Positivo (=1)"])

plt.xlabel("Classi di Sentiment")
plt.ylabel("Numero di Esempi")
plt.title("Bilanciamento delle Classi nel Dataset ridotto e pulito")

# Aggiunta dei valori sopra le barre
for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + 500, str(value), ha='center', fontsize=12)

plt.show()
#%% md
# # Pulizia NLP tweet
#%% md
# ## Conversione del testo in minuscolo
#%%
# Crea una nuova colonna 'text_lower' con il testo convertito in minuscolo
dataset['text_lower'] = dataset['text'].str.lower()

# Visualizza un confronto: prima e dopo la conversione in minuscolo
print("Conversione in Minuscolo (prime 5 righe):\n")
for i, row in dataset[['text', 'text_lower']].head(5).iterrows():
    print("Prima: ", row['text'])
    print("Dopo:  ", row['text_lower'])
    print("-------")
#%% md
# ## Rimozione degli URL
#%%
# Funzione per rimuovere gli URL
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Applica la funzione sul testo in minuscolo
dataset['text_no_url'] = dataset['text_lower'].apply(remove_urls)

# Seleziona solo le righe che contenevano almeno un URL prima della trasformazione
dataset_with_urls = dataset[dataset['text_lower'].str.contains(r'http\S+|www\S+', regex=True, na=False)]

# Visualizza 5 esempi di tweet che contenevano un URL prima della pulizia
print("Rimozione degli URL (5 righe con URL originali):\n")
for i, row in dataset_with_urls[['text_lower', 'text_no_url']].head(5).iterrows():
    print("Prima: ", row['text_lower'])
    print("Dopo:  ", row['text_no_url'])
    print("-------")

#%% md
# ## Rimozione delle menzioni
#%%
# Funzione per rimuovere le menzioni (@username)
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

# Applica la funzione sul testo già privo di URL
dataset['text_no_mentions'] = dataset['text_no_url'].apply(remove_mentions)

# Seleziona solo le righe che contenevano almeno una menzione prima della trasformazione
dataset_with_mentions = dataset[dataset['text_no_url'].str.contains(r'@\w+', regex=True, na=False)]

# Visualizza 5 esempi di tweet che contenevano una menzione prima della pulizia
print("Rimozione delle menzioni (5 righe con menzioni originali):\n")
for i, row in dataset_with_mentions[['text_no_url', 'text_no_mentions']].head(5).iterrows():
    print("Prima: ", row['text_no_url'])
    print("Dopo:  ", row['text_no_mentions'])
    print("-------")
#%% md
# ## Sostituzione delle entità HTML
#%%
# Funzione per sostituire entità HTML:
# - &apos; viene convertito in '
# - Tutte le altre entità (&amp;, &quot;, &lt;, &gt;) vengono rimosse
def replace_html_entities(text):
    text = re.sub(r'&amp;|&quot;|&lt;|&gt;', ' ', text)  # Sostituisci con spazio vuoto
    text = text.replace('&apos;', "'")  # Sostituisci &apos; con apostrofo
    return text

# Applica la funzione sul testo con menzioni già gestite
dataset['text_html_clean'] = dataset['text_no_mentions'].apply(replace_html_entities)

# Filtra solo le righe che contenevano entità HTML nel testo originale
dataset_with_html = dataset[dataset['text_no_mentions'].str.contains(r'&amp;|&quot;|&lt;|&gt;|&apos;', regex=True, na=False)]

# Visualizza 5 esempi: prima (text_no_mentions) e dopo la sostituzione delle entità HTML
print("Sostituzione delle Entità HTML (5 righe con entità HTML):\n")
for i, row in dataset_with_html[['text_no_mentions', 'text_html_clean']].head(5).iterrows():
    print("Prima: ", row['text_no_mentions'])
    print("Dopo:  ", row['text_html_clean'])
    print("-------")
#%% md
# ## Rilevamento emoji testuali e sostituzione con una singola parola
#%%
# Dizionario delle emoji testuali sostituite con una singola parola
emoji_dict = {
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

def replace_text_emojis(text):
    """Sostituisce le emoji testuali con una singola parola."""
    for emoji, meaning in emoji_dict.items():
        text = re.sub(re.escape(emoji), meaning, text)
    return text

# Applica la funzione sul testo pulito senza elementi HTML
dataset['text_no_emojis'] = dataset['text_html_clean'].apply(replace_text_emojis)

# Seleziona solo le righe che contenevano almeno una emoji testuale
dataset_with_emojis = dataset[dataset['text_html_clean'].str.contains('|'.join(map(re.escape, emoji_dict.keys())), regex=True, na=False)]

# Visualizza 5 esempi di tweet che contenevano emoji testuali prima della pulizia
print("Sostituzione delle Emoji Testuali (5 righe con emoji testuali originali):\n")
for i, row in dataset_with_emojis[['text_html_clean', 'text_no_emojis']].head(5).iterrows():
    print("Prima: ", row['text_html_clean'])
    print("Dopo:  ", row['text_no_emojis'])
    print("-------")


#%% md
# ## Rimozione degli hashtag
#%%
# Funzione per gestire gli hashtag: rimuoviamo solo il simbolo '#' mantenendo la parola
def remove_hashtag_symbol(text):
    return re.sub(r'#', '', text)

# Applica la funzione sul testo già privo di emoji testuali
dataset['text_hashtag_clean'] = dataset['text_no_emojis'].apply(remove_hashtag_symbol)

# Seleziona solo le righe che contenevano almeno un hashtag prima della trasformazione
dataset_with_hashtags = dataset[dataset['text_no_emojis'].str.contains(r'#\w+', regex=True, na=False)]

# Visualizza 5 esempi di tweet che contenevano un hashtag prima della pulizia
print("Gestione degli Hashtag (5 righe con Hashtag originali):\n")
for i, row in dataset_with_hashtags[['text_no_emojis', 'text_hashtag_clean']].head(5).iterrows():
    print("Prima: ", row['text_no_emojis'])
    print("Dopo:  ", row['text_hashtag_clean'])
    print("-------")

#%% md
# ## Espansione delle abbreviazioni
#%%
# Creiamo un dizionario di abbreviazioni comuni
abbr_dict = {
    "u": "you", "r": "are", "pls": "please", "idk": "I do not know", "omg": "oh my god",
    "btw": "by the way", "gr8": "great", "lol": "laughing out loud", "brb": "be right back",
    "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing", "smh": "shaking my head",
    "fyi": "for your information", "tbh": "to be honest", "np": "no problem", "imo": "in my opinion",
    "imho": "in my humble opinion", "wtf": "what the fuck", "wth": "what the hell", "gg": "good game",
    "bff": "best friends forever", "ftw": "for the win", "afk": "away from keyboard", "nvm": "never mind",
    "ttyl": "talk to you later", "ikr": "I know, right?", "jk": "just kidding", "idc": "I do not care",
    "hmu": "hit me up", "wyd": "what you doing", "wbu": "what about you", "g2g": "got to go",
    "ty": "thank you", "yw": "you are welcome", "thx": "thanks", "yolo": "you only live once",
    "xoxo": "hugs and kisses", "bday": "birthday", "omw": "on my way", "pov": "point of view",
    "ily": "I love you", "ilu": "I love you", "dm": "direct message", "ggwp": "good game well played",
    "mfw": "my face when", "tfw": "that feeling when", "tldr": "too long did not read", "b4": "before",
    "bc": "because", "b/c": "because", "cuz": "because", "msg": "message", "fml": "fuck my life",
    "fb": "Facebook", "ig": "Instagram", "twt": "Twitter", "faq": "frequently asked questions",
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

# Funzione per espandere le abbreviazioni
def expand_abbreviations(text, abbr_dict):
    tokens = text.split()
    expanded_tokens = [abbr_dict.get(token, token) for token in tokens]
    return " ".join(expanded_tokens)

# Crea una nuova colonna con il testo espanso
dataset['text_abbr_expanded'] = dataset['text_hashtag_clean'].apply(lambda x: expand_abbreviations(x, abbr_dict))

# Filtra solo i tweet che contenevano almeno un'abbreviazione
dataset_abbr_filtered = dataset[dataset['text_hashtag_clean'].apply(lambda x: any(word in abbr_dict for word in x.split()))]

# Visualizza il confronto solo per le righe che hanno subito una modifica
print("Espansione delle Abbreviazioni (solo righe con abbreviazioni):\n")
for i, row in dataset_abbr_filtered[['text_hashtag_clean', 'text_abbr_expanded']].head(10).iterrows():
    print("Prima: ", row['text_hashtag_clean'])
    print("Dopo:  ", row['text_abbr_expanded'])
    print("-------")

#%% md
# ## Espansione delle contrazioni
#%%
# Funzione per espandere le contrazioni usando la libreria
def expand_contractions_lib(text):
    return contractions.fix(text)

# Applica la funzione di espansione
dataset['text_contractions_expanded'] = dataset['text_abbr_expanded'].apply(expand_contractions_lib)

# Seleziona solo le righe in cui il testo è cambiato
changed_rows = dataset[dataset['text_abbr_expanded'] != dataset['text_contractions_expanded']]

# Mostra solo le righe con differenze
print("Espansione delle Contrazioni con Libreria (solo righe modificate):\n")
for i, row in changed_rows[['text_abbr_expanded', 'text_contractions_expanded']].head(10).iterrows():
    print("Prima: ", row['text_abbr_expanded'])
    print("Dopo:  ", row['text_contractions_expanded'])
    print("-------")
#%% md
# ## Rimozione della punteggiatura e caratteri speciali (rimangono solo lettere numeri e spazi)
#%%
# Funzione per rimuovere caratteri speciali (manteniamo solo lettere, numeri e spazi)
def remove_special_chars(text):
    # Modifica: rimuove tutto ciò che non è una lettera (a-z, A-Z) o uno spazio (\s)
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

# Applica la funzione sul testo già pulito dalle contrazioni espanse
dataset['text_no_special'] = dataset['text_contractions_expanded'].apply(remove_special_chars)

# Confronto: prima (text_contractions_expanded) e dopo la rimozione di caratteri speciali e numeri
print("Rimozione di Caratteri Speciali e Numeri (prime 7 righe):\n")
for i, row in dataset[['text_contractions_expanded', 'text_no_special']].head(7).iterrows():
    print("Prima: ", row['text_contractions_expanded'])
    print("Dopo:  ", row['text_no_special'])
    print("-------")
#%% md
# ## Correzione ortografica
#%%
# Parametri per symspellpy
max_edit_distance = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance, prefix_length)

dictionary_path = "../../data/frequency_dictionary_en_82_765.txt"
term_index = 0   # La parola
count_index = 1  # La frequenza

if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
    print("Errore nel caricamento del dizionario!")

def correct_spelling_symspell(text):
    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        # Lookup del token con il livello di verbosità CLOSEST per ottenere la correzione migliore
        suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance)
        if suggestions:
            # Se sono presenti suggerimenti, prendiamo il migliore (primo elemento)
            corrected_tokens.append(suggestions[0].term)
        else:
            # Altrimenti, manteniamo il token originale
            corrected_tokens.append(token)
    return " ".join(corrected_tokens)

# Applica la correzione ortografica sulla colonna ottenuta dalla rimozione dei caratteri speciali e punteggiatura
dataset['text_spell_corrected'] = dataset['text_no_special'].apply(correct_spelling_symspell)

corrections = dataset[dataset['text_no_special'] != dataset['text_spell_corrected']]

# Seleziona un numero limitato di esempi per la visualizzazione
num_examples = 20
corrections_sample = corrections[['text_no_special', 'text_spell_corrected']].head(num_examples)

# Stampa le correzioni effettive
print("Esempi di correzioni ortografiche effettuate da SymSpell:\n")
for i, row in corrections_sample.iterrows():
    print("Prima: ", row['text_no_special'])
    print("Dopo:  ", row['text_spell_corrected'])
    print("-------")

#%% md
# ## Normalizzazione spazi
#%%
def normalize_spaces(text):
    # Rimuove spazi multipli e spazi iniziali/finali
    return re.sub(r'\s+', ' ', text).strip()

# Crea una nuova colonna
dataset['text_spaces_clean'] = dataset['text_spell_corrected'].apply(normalize_spaces)

# Confronto (prime 5 righe)
print("Rimozione spazi multipli e trimming:\n")
for i, row in dataset[['text_spell_corrected', 'text_spaces_clean']].head(5).iterrows():
    print("PRIMA ->", row['text_spell_corrected'])
    print("DOPO  ->", row['text_spaces_clean'])
    print("-------------------------------------------------")
#%% md
# ## Tokenizzazione, lemmatizzazione e rimozione delle stopwords
#%%
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenizzazione: divide il testo in parole
    tokens = word_tokenize(text)

    # Lemmatizzazione: ottiene la forma base di ogni token
    tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

    # Rimozione delle stopword: elimina i token che sono presenti nella lista
    tokens_clean = [token for token in tokens_lemmatized if token.lower() not in stop_words]

    return tokens_clean

# Applica il preprocessing su una nuova colonna
dataset['tokens_processed'] = dataset['text_spaces_clean'].apply(preprocess_text)

# Per il confronto, mostra il risultato per alcune righe
print("\nConfronto prima e dopo (Tokenizzazione, Lemmatizzazione e Rimozione Stopword):\n")
for i, row in dataset[['text_spaces_clean', 'tokens_processed']].head(5).iterrows():
    print("Testo originale:", row['text_spaces_clean'])
    print("Token elaborati:", row['tokens_processed'])
    print("-------------------------------------------------")
#%% md
# ## Ricostruzione del testo
#%%
# Funzione per ricostruire il testo dai token processati
def reconstruct_text(tokens):
    return " ".join(tokens)

# Crea una nuova colonna con il testo ricostruito
dataset['text_final'] = dataset['tokens_processed'].apply(reconstruct_text)

# Visualizza il confronto tra il testo tokenizzato e il testo ricostruito
print("Confronto tra il testo originale e il testo ricostruito:\n")
for i, row in dataset[['tokens_processed', 'text_final']].head(5).iterrows():
    print("Prima: ", row['tokens_processed'])
    print("Dopo:  ", row['text_final'])
    print("-------")
#%% md
# ## Analisi word cloud sentiment positivo e negativo
#%%
# Seleziona tutti i tweet con target 0 (sentiment negativo) e unisci il testo in un'unica stringa
text_neg = ' '.join(dataset[dataset['target'] == 0]['text_final'])

# Genera la wordcloud per il sentiment negativo
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text_neg)

# Visualizza la wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Parole Frequenti - Sentiment Negativo')
plt.axis('off')
plt.show()
#%%
# Seleziona tutti i tweet con target 1 (sentiment positivo) e unisci il testo in un'unica stringa
text_pos = ' '.join(dataset[dataset['target'] == 1]['text_final'])

# Genera la wordcloud per il sentiment positivo
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text_pos)

# Visualizza la wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Parole Frequenti - Sentiment Positivo')
plt.axis('off')
plt.show()
#%% md
# ## Rimozione parola "wa"
#%%
def remove_wa(text):
    return re.sub(r'\bwa\b', '', text)

# Applica la funzione di rimozione della parola "wa"
dataset['text_final_no_wa'] = dataset['text_final'].apply(remove_wa)

# Normalizza nuovamente gli spazi per rimuovere eventuali spazi extra dopo la rimozione
def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

dataset['text_final_no_wa'] = dataset['text_final_no_wa'].apply(normalize_spaces)

# Filtra il dataset per ottenere solo i tweet che contenevano "wa" prima della rimozione
mask = dataset['text_final'].str.contains(r'\bwa\b', regex=True)

# Seleziona solo le righe che soddisfano la condizione e le colonne per il confronto
df_with_wa = dataset.loc[mask, ['text_final', 'text_final_no_wa']]

# Confronto finale
print("Confronto finale per i tweet che contenevano la parola 'wa':\n")
for i, row in df_with_wa.iterrows():
    print("PRIMA ->", row['text_final'])
    print("DOPO  ->", row['text_final_no_wa'])
    print("-------------------------------------------------")

#%% md
# ## Analisi word cloud sentiment positivo e negativo
#%%
# Seleziona tutti i tweet con target 0 (sentiment negativo) e unisci il testo in un'unica stringa
text_neg = ' '.join(dataset[dataset['target'] == 0]['text_final_no_wa'])

# Genera la wordcloud per il sentiment negativo
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text_neg)

# Visualizza la wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Parole Frequenti - Sentiment Negativo')
plt.axis('off')
plt.show()
#%%
# Seleziona tutti i tweet con target 1 (sentiment positivo) e unisci il testo in un'unica stringa
text_pos = ' '.join(dataset[dataset['target'] == 1]['text_final_no_wa'])

# Genera la wordcloud per il sentiment positivo
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text_pos)

# Visualizza la wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Parole Frequenti - Sentiment Positivo')
plt.axis('off')
plt.show()
#%%
#Verifica di eventuali tweet vuoti
empty_tweets = dataset[dataset['text_final_no_wa'] == '']
print("Numero di tweet vuoti:", empty_tweets.shape[0])

#Rimozione dei tweet vuoti
dataset = dataset[dataset['text_final_no_wa'] != '']

dataset
#%% md
# ## Applicazione Tf-Idf
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# Inizializza il vettorizzatore TF-IDF.
tfidf = TfidfVectorizer(min_df=5, max_df=0.9)

# Applica TF-IDF sulla colonna 'text_final_no_wa'
X_sparse = tfidf.fit_transform(dataset['text_final_no_wa'])

# Stampa della matrice sparsa: vengono mostrati il tipo e la shape
print("Matrice TF-IDF (sparsa):")
print(X_sparse)
print("\nShape della matrice sparsa:", X_sparse.shape)

# Converte la matrice sparsa in una matrice densa e stampa il risultato
X_dense = X_sparse.todense()
print("\nMatrice TF-IDF (densa):")
print(X_dense)

# Ottiene le feature names (vocaboli) e stampa le prime 10 per far capire quali token sono stati estratti
feature_names = tfidf.get_feature_names_out()
print("\nPrime 30 feature names:")
print(feature_names[:30])