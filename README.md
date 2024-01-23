# Projet de Classification et Indexation d'Images

## Objectif du Projet
Ce projet vise à créer un modèle de Deep Learning pour la classification d'images en se basant sur les descripteurs extraits à l'aide de l'algorithme HoG (Histogram of Oriented Gradients). Par la suite, l'application de l'algorithme CNN (Convolutional Neural Network) est utilisée pour la classification des images. Le but final est l'indexation des images en se basant sur le contenu et la similarité entre les vecteurs descripteurs, en utilisant l'algorithme KNN (K-Nearest Neighbors) pour calculer les distances entre les vecteurs.

## Structure du Projet
Le projet est structuré comme suit:

- **data/ :** Ce répertoire contient les ensembles de données utilisés pour l'entraînement et les tests du modèle.
- **src/ :** Les fichiers source du projet, comprenant l'implémentation de l'algorithme HoG, le modèle CNN, et le code pour l'application de KNN.
- **models/ :** Ce répertoire stocke les modèles entraînés.
- **results/ :** Les résultats de l'évaluation du modèle et les images indexées.

## Configuration et Prérequis
Assurez-vous d'avoir les dépendances suivantes installées:

- Python 3.x
- Bibliothèques Python telles que TensorFlow, OpenCV, scikit-learn, etc. (vous pouvez les installer en utilisant `pip install -r requirements.txt`)

## Utilisation
1. **Extraction des Descripteurs HoG :**
   - Exécutez le script `extract_hog_descriptors.py` pour extraire les descripteurs HoG à partir des images.

2. **Entraînement du Modèle CNN :**
   - Utilisez le script `train_cnn_model.py` pour entraîner le modèle CNN en utilisant les descripteurs HoG.

3. **Classification des Images :**
   - Appliquez la classification des images à l'aide du script `classify_images.py`.

4. **Indexation des Images :**
   - Utilisez le script `index_images.py` pour indexer les images en calculant la similarité entre les vecteurs descripteurs.

## Résultats
Les résultats de l'entraînement du modèle, de la classification des images, et de l'indexation sont stockés dans le répertoire `results/`.

