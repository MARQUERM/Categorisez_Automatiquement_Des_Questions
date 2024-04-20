# Projet: Prédiction de tags pour [StackOverflow](https://stackoverflow.com/)

Ce projet vise à créer une API de prédiction de tags basée sur différents modèles d'apprentissage automatique. Plusieurs modèles, y compris le Universal Sentence Encoder (USE) de TensorFlow, ont été évalués. Finalement, le modèle retenu pour l'API est basé sur la vectorisation TF-IDF avec FastAPI.

L'objectif principal est de prédire des tags pertinents pour les questions posées par les utilisateurs.

# Notebook
Notebook 1: partie exploration et nettoyage des données
Notebook 2: utilisation de StackAPI pour la recupération de 50 questions
Notebook 3: approche non supervisée avec utilisation de pyLDAvis
Notebook 4: approche supervisée
Notebook 5: code API
Notebook 6: point d'entrée pour l'API

# Installation
Projet fonctionnant sous python 3.10

Requirements:
pip install -r requirements.txt


# Exécution

Sur le notebook 4: démarrer MLFlow avec la commande "mlflow ui" dans le terminal

Pour les tests unitaire:
pytest tests/

Pour l'API:
Lancez l'API avec la commande suivante:
uvicorn main:app 


Heroku:
Demarrage sur: https://dashboard.heroku.com/apps/stacktags/resources

# Lien
URL de l'API: https://stacktags-fc3e30f462b9.herokuapp.com/predict


