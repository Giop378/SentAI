import pandas as pd
#Caricamento e visualizzazione del dataset
dataset = pd.read_csv("../../data/sentiment140_remaining.csv", encoding="ISO-8859-1")
dataset = dataset[['text', 'target']]
dataset['target'] = dataset['target'].apply(lambda x: 1 if x == 4 else 0)
dataset = dataset[dataset['text'].str.len() <= 140]
dataset.to_csv("../../data/sentiment140_remaining_clean.csv", index=False)
dataset = pd.read_csv("../../data/sentiment140_remaining_clean.csv")
print(dataset.shape)