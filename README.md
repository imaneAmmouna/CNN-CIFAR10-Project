# 🌟 CNN Introduction Project — CIFAR-10 Classification

Ce projet constitue ma première expérience pratique avec les concepts de **Machine Learning (ML)** et de **Deep Learning (DL)** en utilisant **TensorFlow** et **Keras**.  
Il se concentre sur la création et l’entraînement d’un **réseau de neurones convolutif (CNN)** destiné à la **classification d’images** issues du jeu de données **CIFAR-10**.

---

## Objectives
Ce projet met en œuvre un réseau de neurones convolutif (CNN) construit avec TensorFlow/Keras pour la classification d’images issues du jeu de données CIFAR-10.
L’objectif est de reconnaître automatiquement les objets présents dans des images couleur 32x32 appartenant à 10 classes (avion, voiture, chat, chien, etc.).

Le notebook couvre toutes les étapes du pipeline d’apprentissage profond :
- Chargement et exploration du dataset
- Prétraitement et visualisation
- Construction d’un CNN optimisé avec Batch Normalization et Dropout
- Data Augmentation pour éviter le surapprentissage
- Entraînement, évaluation et visualisation des résultats
- Prédiction et interprétation des résultats avec la matrice de confusion et des exemples visuels

---

## Dataset : **CIFAR-10**
Source : https://www.cs.toronto.edu/~kriz/cifar.html
Classes (10) :
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
Taille : 60 000 images couleur (32x32 pixels)
- 50 000 pour l’entraînement
- 10 000 pour le test

---

## Data Preprocessing

Avant d’entraîner le modèle CNN, il est essentiel de préparer et transformer les données pour garantir une meilleure performance.

### Normalisation des images
Les pixels des images CIFAR-10 sont initialement codés de 0 à 255.  
Pour que le réseau de neurones converge plus rapidement et efficacement, les valeurs sont normalisées dans l’intervalle `[0, 1]`.  
Cette étape permet au modèle d’apprendre plus efficacement et d’éviter que certaines valeurs dominent l’apprentissage.

### Encodage des labels (One-Hot Encoding)
Les étiquettes du dataset sont des entiers de 0 à 9 représentant les 10 classes.  
Pour la classification multi-classes, les labels sont transformés en **vecteurs one-hot**, où chaque classe est représentée par un vecteur dont la position correspondant à la classe est 1 et toutes les autres positions sont 0.  
Par exemple, la classe “chat” devient un vecteur où seule la quatrième position est à 1, et toutes les autres sont à 0.  

---

## Architecture du modèle

Le modèle utilisé est un **réseau de neurones convolutif (CNN)** conçu pour la classification d’images.  
Le CNN est constitué de plusieurs blocs successifs :
  - **Convolution (Conv2D)** : extrait automatiquement les caractéristiques importantes des images (bords, textures, formes).
  - **Batch Normalization** : stabilise et accélère l’apprentissage en normalisant les activations de chaque couche.
  - **Max Pooling** : réduit la dimension des cartes de caractéristiques et conserve les informations les plus importantes.
  - **Dropout** : réduit le surapprentissage en désactivant aléatoirement certaines neurones pendant l’entraînement.

- Trois blocs de convolution sont utilisés avec un nombre de filtres croissant (32 → 64 → 128) pour capturer des caractéristiques de plus en plus complexes.

Couche Fully Connected (Dense) :
Après les blocs convolutionnels et la mise à plat (Flatten) :
- Une couche dense avec 128 neurones et activation ReLU permet de combiner les caractéristiques extraites.
- Une couche finale dense avec 10 neurones et activation softmax produit la **probabilité de chaque classe** pour l’image.

Techniques pour améliorer les performances :
- **Dropout** : réduit le surapprentissage en désactivant aléatoirement certaines connexions.
- **Batch Normalization** : améliore la stabilité et la vitesse d’apprentissage.
- **Data Augmentation** (utilisée lors de l’entraînement) : génère des variations d’images (translations, rotations, flips horizontaux) pour rendre le modèle plus robuste.

---

## Entraînement et Évaluation

### Compilation du modèle
Le modèle CNN est compilé avec les paramètres suivants :
- **Fonction de perte (Loss function)** : `categorical_crossentropy` pour les problèmes de classification multi-classes.
- **Optimiseur** : `Adam`, qui ajuste automatiquement le taux d’apprentissage pour une convergence rapide.
- **Métriques suivies** :
  - **Accuracy** : pourcentage de prédictions correctes.
  - **Precision (Précision)** : proportion des prédictions positives correctes.
  - **Recall (Rappel / Sensibilité)** : proportion des vraies classes positives correctement identifiées.

### Early Stopping
- Pour éviter le surapprentissage (overfitting), une technique d’**early stopping** peut être utilisée, arrêtant l’entraînement lorsque la performance sur les données de validation ne s’améliore plus après quelques itérations.

### Courbes d’apprentissage
- **Loss et validation loss** : suivent la fonction de perte pour détecter le surapprentissage.
- **Accuracy et validation accuracy** : indiquent la performance globale du modèle.
- **Precision et Recall** : permettent d’évaluer la qualité des prédictions pour chaque classe.

### Évaluation finale
- Le modèle est évalué sur le **jeu de test** (10 000 images).
- Une **matrice de confusion** est générée pour visualiser les performances par classe.
- Un **rapport de classification** fournit précision, rappel et F1-score pour chacune des 10 classes CIFAR-10.

### Visualisation des résultats
- Des images de test sont affichées avec les **prédictions du modèle**.
- Les bonnes prédictions sont en bleu, les erreurs en rouge.
- Des graphiques montrent la **distribution des probabilités** prédites pour chaque image.

---

## Résultats de l’entraînement

Après l’entraînement du modèle CNN sur le dataset CIFAR-10, voici les résultats obtenus :

### Performance globale
- **Accuracy sur le jeu de test** : ~85%
- **Précision moyenne (Macro Precision)** : 85%
- **Rappel moyen (Macro Recall)** : 85%
- **F1-score moyen** : 85%

### Rapport de classification par classe
| Classe       | Précision | Rappel | F1-score |
|--------------|-----------|--------|----------|
| Airplane     | 0.86      | 0.88   | 0.87     |
| Automobile   | 0.96      | 0.93   | 0.94     |
| Bird         | 0.83      | 0.75   | 0.79     |
| Cat          | 0.74      | 0.71   | 0.73     |
| Deer         | 0.84      | 0.82   | 0.83     |
| Dog          | 0.88      | 0.70   | 0.78     |
| Frog         | 0.74      | 0.95   | 0.83     |
| Horse        | 0.90      | 0.89   | 0.89     |
| Ship         | 0.91      | 0.92   | 0.92     |
| Truck        | 0.88      | 0.95   | 0.92     |

### Visualisation des prédictions
- Le modèle prédit correctement la majorité des images du test set.
- Les bonnes prédictions sont visualisées en **bleu**, et les erreurs en **rouge**.
- Chaque prédiction est accompagnée d’un graphique montrant les probabilités pour toutes les classes, ce qui permet d’analyser la confiance du modèle.

> Ces résultats montrent que le modèle CNN, combiné avec la normalisation, le one-hot encoding et la data augmentation, est capable de classer les images CIFAR-10 avec une bonne précision et robustesse.
