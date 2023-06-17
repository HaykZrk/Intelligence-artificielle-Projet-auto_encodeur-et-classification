import numpy as np

class Couche:
  def __init__(self, dim_entree, dim_sortie, activation):
    self.entree = None
    self.sortie = None
    self.poids = (np.random.rand(dim_entree, dim_sortie) - 0.5)
    self.biais = (np.random.rand(1, dim_sortie) - 0.5)
    self.activation = activation

  def propagation_avant(self, donnee_entree):
    """ Calcule la sortie d'une couche de neurones en fonction des 
    entrées, des poids, des biais et de la fonction d'activation. """
    self.entree = donnee_entree
    Z = np.matmul(self.entree, self.poids) + self.biais
    self.sortie = self.activation(Z)[0]
    return self.sortie

  def propagation_arriere(self, E_Y, eta):
    """ Met à jour les poids et les biais à partir de dE/dY et 
    retourne dE/dX """
    # Calcul de l'erreur avant l'activation
    E_Y = self.activation(self.sortie)[1] * E_Y

    E_X = np.matmul(E_Y, self.poids.T)
    E_W = np.matmul(self.entree.T, E_Y)
    E_B = E_Y
    self.poids -= eta * E_W
    self.biais -= eta * E_B

    return E_X