<<<<<<< HEAD
"""Visualisations"""
=======
# visualize.py — visualisation des images et des features
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
from sklearn.manifold import TSNE

def show_random_samples(df, n=9, img_size=(224,224)):
    """Affiche des échantillons aléatoires du dataset."""
    samples = df.sample(n)
    cols = int(math.sqrt(n))
    plt.figure(figsize=(12,8))
    for i, row in samples.iterrows():
        img = Image.open(row["path"]).convert("RGB").resize(img_size)
        plt.subplot(math.ceil(n/cols), cols, i+1)
        plt.imshow(img)
        plt.title(row["label"])
        plt.axis("off")
    plt.show()

def plot_class_distribution(df):
    df["label"].value_counts().plot(kind="bar", title="Répartition des classes")

def plot_tsne_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30)
    X2 = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(X2[idx,0], X2[idx,1], label=lbl, alpha=0.7)
    plt.legend()
    plt.title("t-SNE des embeddings")
    plt.show()
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
