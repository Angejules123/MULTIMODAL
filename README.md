# IA médicale multimodale & explicable (XAI)

> Système d'aide au diagnostic précoce de troubles cognitifs, combinant **signaux EEG**,
> **imagerie IRM** et **données d'activités de la vie quotidienne**, avec une contrainte
> centrale : chaque prédiction doit pouvoir être justifiée.

Projet académique — Master 2 Science des Données, 2025. Mené en binôme.

**Troubles couverts :** Alzheimer, TDAH, autisme, dépression, stress chronique.

---

## Pourquoi l'explicabilité d'abord

En santé, un modèle qui annonce un diagnostic sans pouvoir dire *pourquoi* n'a aucune
valeur décisionnelle. Un clinicien ne peut ni le contredire, ni le suivre en conscience,
ni en rendre compte au patient. L'explicabilité n'est donc pas ici un supplément de
confort : c'est ce qui détermine si le système est utilisable ou non.

C'est pourquoi l'approche XAI structure l'ensemble du projet plutôt que d'être ajoutée
en fin de chaîne.

## Pourquoi le multimodal

Aucune des trois sources ne suffit isolément :

| Modalité | Ce qu'elle apporte | Sa limite |
|---|---|---|
| EEG | Activité électrique, dynamique temporelle | Bruité, sensible aux artefacts |
| IRM | Structure cérébrale, lésions | Statique, coûteuse, peu répétable |
| Activités quotidiennes | Retentissement fonctionnel réel | Subjective, déclarative |

Les combiner permet de croiser un signal physiologique, une image structurelle et une
observation comportementale — trois angles sur le même phénomène.

## Approche

- **Classification d'images médicales** : ResNet et architectures personnalisées.
- **Traitement des signaux EEG** : prétraitement, extraction de caractéristiques.
- **Fusion multimodale** des trois sources.
- **Démonstrateur interactif** développé avec Streamlit, permettant de charger un cas et
  de visualiser la prédiction accompagnée de ses éléments justificatifs.

## Stack

Python · PyTorch · ResNet · Streamlit · TensorBoard

## Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Ajuster selon l'arborescence réelle du dépôt.

## Limites et suites possibles

Ce travail est un **prototype de recherche**, en aucun cas un dispositif médical. Il n'a
pas fait l'objet d'une validation clinique et ne doit pas être utilisé à des fins de
diagnostic réel. Les pistes d'amélioration identifiées portent sur la taille et la
diversité des jeux de données, et sur une évaluation par des cliniciens.

---

**Ange-Jules TIA** — tangejules@gmail.com · [LinkedIn](https://www.linkedin.com/in/ange-jules-tia-1a7b4a220)
