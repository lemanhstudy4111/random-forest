import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle


class Bootstrap:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)
    
    def sample(self):
        rng = np.random.default_rng()
        indices = rng.choice(self.dataset.shape[0], size=self.dataset.shape[0], replace=True)
        return self.dataset[indices]

class StratifiedValidation:
    def __init__(self, dataset, k=5):
        self.dataset = dataset
        self.pos = np.where(dataset["label"] == 1)[0]
        self.neg = np.where(dataset["label"] == 0)[0]
        self.k = k
    
    def kfold_split(self):
        rng = np.random.default_rng()
        pos_indices = rng.permutation(self.pos)
        neg_indices = rng.permutation(self.neg)

        pos_folds = np.array_split(pos_indices, self.k)
        neg_folds = np.array_split(neg_indices, self.k)

        folds = []
        for i in range(self.k):
            fold_indices = np.concatenate((pos_folds[i], neg_folds[i]))
            fold_indices = rng.permutation(fold_indices)
            folds.append(self.dataset[fold_indices])
        
        return folds

class Performance:
    def __init__(self, y=None, preds=None):
        self.y = np.array(y)
        self.preds = np.array(preds)
        self.precision = -1
        self.recall = -1
    
    def get_accuracy(self):
        return np.sum(self.preds == self.y)/self.y.shape[0]
    
    def get_precision(self):
        _, preds_counts = np.unique(self.preds, return_counts=True)
        tp = np.sum(np.logical_and(self.y == 1, self.preds == 1))
        self.precision = tp / preds_counts[1]
        return self.precision
    
    def get_recall(self):
        _, actual_counts = np.unique(self.y, return_counts=True)
        tp = np.sum(np.logical_and(self.y == 1, self.preds == 1))
        self.recall = tp / actual_counts[1]
        return self.recall
    
    def get_f1(self):
        return 2*((self.precision * self.recall)/(self.precision + self.recall))