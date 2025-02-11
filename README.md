# ![SentAI-Logo](https://github.com/user-attachments/assets/d8236502-9053-420e-9808-8464491f41f9)
# SentAI

Il progetto **SentAI** ha lo scopo di sviluppare un modello in grado di analizzare testi in inglese brevi e generici e determinarne il sentimento (positivo o negativo). Nello specifico, utilizza due classificatori:
- **Naive Bayes**
- **Logistic Regression**

L’intero flusso segue il modello **CRISP-DM** (Cross Industry Standard Process for Data Mining), che comprende le seguenti fasi:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## Requisiti

Assicurati di aver installato tutte le librerie necessarie eseguendo:

```bash
pip install -r requirements.txt
```
## Preparazione del Dataset
Scarica il dataset Sentiment140 da Kaggle al link: https://www.kaggle.com/datasets/kazanova/sentiment140

Dopo aver scaricato il file CSV, posizionalo nella cartella data del progetto e rinominalo in:
```bash
data/sentiment140_raw.csv
```
## Data Preparation
Per creare i dataset necessari, esegui i seguenti script:
1. Avvia la prima fase di preparazione dei dati:
   ```bash
python src/02_data_preparation/data_preparation.py
```
2. Completa la preparazione del dataset eseguendo:
   ```bash
python src/02_data_preparation/preparation_sentiment140_remaining.py
```

## Modeling
Per addestrare il modello con Logistic Regression e TF-IDF, esegui il seguente script:
```bash
python src/03_modeling/modeling_logistic_regression_tf_idf.py
```
Il modello verrà salvato in:
```bash
src/03_modeling/logistic_regression_tfidf_pipeline.joblib
```
## Evaluation
Per valutare il modello sul restante dataset, esegui il seguente script:
```bash
python src/04_evaluation/evaluation_remaining_dataset.py
```

## Deployment
Per effettuare il deployment del modello ed eseguire l’interfaccia grafica, lancia il seguente comando:
```bash
python src/05_deployment/deployment.py
```
Questo script caricherà il modello allenato e avvierà un server locale per testare l’applicazione con una semplice interfaccia grafica.
