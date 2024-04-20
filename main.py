from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fonction import clean_and_filter, transform_bow_lem_fct
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI()

# Chargement du modèle TF-IDF
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Chargement du modèle LogisticRegression
logisticRegression_model = joblib.load("logisticRegression_model.joblib")

# Noms des tags
mlb_classes = joblib.load('mlb_classes.pkl')

class PredictionInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text: str
    predicted_tags: list

@app.post("/predict")
def predict_tags(payload: PredictionInput):
    # Prétraitement
    preprocessed_text = clean_and_filter(payload.text)
    preprocessed_text = transform_bow_lem_fct(preprocessed_text)

    # Utilisation du modèle TF-IDF
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Prédiction avec le modèle LogisticRegression
    predicted_tags = logisticRegression_model.predict(text_tfidf)

    # Indices où les tags sont prédits (1)
    predicted_tags_indices = np.where(predicted_tags[0] == 1)[0]

    # Noms des tags prédits
    predicted_tags_names = mlb_classes[predicted_tags_indices]

    # Résultats
    result = PredictionOutput(text=payload.text, predicted_tags=predicted_tags_names)
    return result