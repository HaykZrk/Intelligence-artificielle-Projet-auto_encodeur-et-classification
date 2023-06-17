""" Auteurs : Alexandre DUBERT et Hayk ZARIKIAN (TP6) """

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from neurones.utility import tanh, MSE_cost, d_MSE_cost
from neurones.couche import Couche
from neurones.reseau import Reseau
from arbre.arbre import Arbre

def visualisation(data):
  """ Affiche les densités de probabilité selon chaque attribut pour tous les
  price_range. """
  headers = data.columns[1:-1]
  for h in headers:
    sns.kdeplot(data=data, x=h, hue='price_range')
    plt.show()

def normalisation(data):
  """ Normalise les données entre -1 et 1. """
  cp = data.copy()
    
  col_min = data.min()
  col_max = data.max()
    
  headers = data.columns[1:-1]
  for h in headers:
    cp[h] = (data[h] - col_min[h]) / (col_max[h] - col_min[h]) * 2 - 1
    
  return cp

def coller_etiquette(data,etiquette):
  """ attache l'étiquette correspondante à chaque ligne. """
  res_list = []
  for i in range(len(data)):
    row = []
    row_d = data[i].tolist()[0]
    for k in range(len(row_d)):
      row.append(row_d[k])
    row.append(etiquette[i])
    res_list.append(row)
  return res_list

# Initialisation des données
data_rep = './data/'
df = pd.read_csv(data_rep + "raw_train.csv")
# visualisation(df)

# Préparation du jeu de données
norm_df = normalisation(df)
x_train = norm_df.iloc[: , 1:-1].to_numpy()
y_train = x_train

# Initialisation du réseau
res = Reseau()
res.add(Couche(20, 2, tanh))
res.add(Couche(2, 20, tanh))
res.use(MSE_cost, d_MSE_cost)

# Entraînement et test de l'auto-encodeur
res.fit(x_train, y_train, 100, 0.01)
# y_chap = res.predict(x_train)

# Encodage des données
donnee_encode = res.encodage(x_train)
donnee_encode = coller_etiquette(donnee_encode, df['price_range'])
attr_encodes = ['attr_code1','attr_code2','price_range']
df_encode = pd.DataFrame(donnee_encode, columns=attr_encodes)

# Initialisation de l'arbre et inférence des classes
arbre = Arbre(df_encode, 'price_range', attr_encodes[:-1])
arbre.predire()