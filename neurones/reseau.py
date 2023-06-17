import numpy as np

class Reseau:
  def __init__(self):
    self.couches = []
    self.cout = None
    self.d_cout = None

  def add(self, couche):
    """ Ajout d'une nouvelle couche au réseau. """
    self.couches.append(couche)

  # set cout to use
  def use(self, cout, d_cout):
    """ Choix de la fonction de coût du réseau. """
    self.cout = cout
    self.d_cout = d_cout

  def predict(self, input_data):
    """ Prédit Ŷ en fonction des données X. """
    # sample dimension first
    samples = len(input_data)
    result = []

    # run network over all samples
    for i in range(samples):
      # forward propagation
      output = input_data[i]
      for couche in self.couches:
        output = couche.propagation_avant(output)
      result.append(output)
    return result

  def fit(self, x_train, y_train, epochs, learning_rate):
    """ Entraîne le réseau en effectuant des propagations en avant et en 
    arrière pour chaque ligne de données, pendant epochs d'itérations. """
    # sample dimension first
    samples = len(x_train)

    # training loop
    for i in range(epochs):
      err = 0
      for j in range(samples):
        # forward propagation
        output = np.array([x_train[j]])
        for couche in self.couches:
          output = couche.propagation_avant(output)

        # compute cout (for display purpose only)
        err += self.cout(y_train[j], output)

        # backward propagation
        error = self.d_cout(y_train[j], output)
        for couche in reversed(self.couches):
          error = couche.propagation_arriere(error, learning_rate)

      # calculate average error on all samples
      err /= samples
      if i % 10 == 0:
        print('epoch %d/%d   error=%f' % (i+1, epochs, err))

  def encodage(self, input_data):
    """ Encode les données en entrée pour rétrécir la dimension. """
    code = []
    samples = len(input_data)

    for i in range(samples):
      row = input_data[i]
      c1 = self.couches[0]
      code.append(c1.propagation_avant(row))

    return code
