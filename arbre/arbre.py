import math

class Noeud:

  def __init__(self, 
               val = None, 
               attribut = None, 
               fils_gauche = None, 
               fils_droit = None, 
               prediction = None, 
               feuille = False):
    self.val = val
    self.attribut = attribut
    self.fils_gauche = fils_gauche
    self.fils_droit = fils_droit
    self.prediction = prediction
    self.feuille = feuille
   
  def isLeaf(self):
    if self.fils_gauche is None and self.fils_droit is None:
      return True

  def __str__(self):
    return "<" + str(self.val) + " " + str(self.fils_gauche) + " " + \
      str(self.fils_droit) + ">"

  def node_result(self, spacing = ' '):
    s = ''
    for v in range(len(self.prediction.values)):
      s += 'class' + str(self.prediction.index[v]) +  \
           ' count = ' + str(self.prediction.values[v]) + \
           '\n' + spacing
    return s

class Arbre:

  def __init__(self, data, etiquette, attributs):
    self.data = data
    self.etiquette = etiquette
    self.attributs = attributs
    self.racine = self.construction_arbre(0)

  def meilleur_attribut(self):
    best_attribut = None
    best_gain = -1 #Q: Technique "sale", voir comment ça peut marcher mieux
    best_split = None
    best_partitions = None

    for attr in self.attributs:
      attribut, gain, split, partitions = \
        gain_split_partition(self.data, self.etiquette, attr)

      if gain > best_gain:
        best_attribut = attribut
        best_gain = gain
        best_split = split
        best_partitions = partitions

    return (best_attribut, best_gain, best_split, best_partitions)

  def construction_arbre(self, profondeur):
    attribut, gain, split, partitions \
     = self.meilleur_attribut()
    prediction = self.data[self.etiquette].value_counts()
    
    #AJG : Gain == 0, permet de trier les noeuds purs
    if((profondeur > 2) or (len(self.attributs) == 0) or (gain == 0)): 
      return Noeud(
        val = split, 
        attribut = attribut, 
        prediction = prediction, 
        feuille = True
      )

    self.attributs.remove(attribut)
    branche_gauche = self.construction_arbre(profondeur+1)
    # print(branche_gauche)
    branche_droite = self.construction_arbre(profondeur+1)
    # print(branche_droite)

    return Noeud(split, attribut, branche_gauche, branche_droite, prediction)

  def print_tree(self, noeud, spacing=''):
    if noeud is None:
      return
    if noeud.isLeaf():
      print(spacing + noeud.node_result(spacing))
      return

    print('{}[ Attribute: {} Split value: {}]'.
      format(spacing, noeud.attribut, noeud.val))

    print(spacing + '> True ')
    self.print_tree(noeud.fils_gauche, spacing + '-')

    print(spacing + '> False')
    self.print_tree(noeud.fils_droit, spacing + '-')
    return

  def afficher(self):
    self.print_tree(self.racine)

  def inference(self, instance, noeud):
    val_attribut = 0
    if noeud.isLeaf():
      print(noeud.prediction)
    else:
      val_attribut = instance[noeud.attribut]
      if(val_attribut < noeud.val):
        self.inference(instance, noeud.fils_gauche)
      else:
        self.inference(instance, noeud.fils_droit)

  def predire(self):
    for i in range(len(self.data)):
      self.inference(self.data.iloc[i], self.racine)

def entropie(dataframe, etiquette):
    nb_lignes = dataframe.shape[0]
    series = dataframe[etiquette].value_counts()
    entropie = 0
    for valeur in series:
        proportion = valeur / nb_lignes
        entropie -= proportion * math.log2(proportion)
    return entropie

def gain_split_partition(dataframe, etiquette, attribut):
  H = entropie(dataframe, etiquette)
  gain = 0
  split_value = 0
  partitions = [None, None]
  data_sorted = dataframe.sort_values(by = attribut)
  sum = 0
  attribut_a = None

  for i in range(len(data_sorted)):
    attribut_a = data_sorted[attribut].iloc[i]
    if(data_sorted[etiquette].iloc[i] != data_sorted[etiquette].iloc[0]):
      split_value = attribut_a
      # données pour lesquelles a < split_value
      partitions[0] = data_sorted[data_sorted[attribut] < split_value] 
      # données pour lesquelles a >= split_value
      partitions[1] = data_sorted[data_sorted[attribut] >= split_value] 
    
      for p in partitions:
        sum += (len(p) / len(dataframe)) * entropie(p, etiquette)
      gain = H - sum 

      break

  return (attribut, gain, split_value, partitions)