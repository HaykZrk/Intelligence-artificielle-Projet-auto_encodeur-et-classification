import numpy as np

def tanh(Z):
  """
  Z : non activated outputs
  Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
  """
  #A = np.empty(Z.shape)
  A = np.tanh(Z)
  df = 1-A**2
  return A,df
    
def MSE_cost(y_hat, y):
  """ Calcul de l'erreur quadratique moyenne """
  mse = np.square(np.subtract(y_hat, y)).mean()
  return mse

def d_MSE_cost(y_hat, y):
  """ Calcul de la dérivée de l'erreur quadratique moyenne """
  dmse = 2*np.subtract(y, y_hat)/y_hat.size
  return dmse