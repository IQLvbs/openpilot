import numpy as np


def safe_exp(values, out=None):
  return np.exp(np.clip(values, -np.inf, 11), out=out)


def sigmoid(values):
  return 1.0 / (1.0 + safe_exp(-values))
