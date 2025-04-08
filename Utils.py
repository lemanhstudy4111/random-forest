import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Bootstrap:
    def __init__(self, filename):
        self.shape = None
        self.filename = filename
        self.dataset = np.genfromtxt(self.filename, delimiter=',')
    
    def sampling(self):
        return np.random.Generator.choice(self.dataset, size=self.shape, replace=True)
    
class Performance:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
        self.df = None
        self.df_confusion = None
    
    def create_eval_df(self):
        df_pos = pd.DataFrame({"doc": self.pos, "actual": "P"})
        df_neg = pd.DataFrame({"doc": self.neg, "actual": "N"})
        self.df = pd.concat([df_pos, df_neg])
        return
    
    def add_prediction(self, predicts):
        self.df["predicted"] = predicts
        
    def reset_prediction(self):
        self.df.drop(columns=["predicted"], inplace=True)
    
    #df_predict: 2 columns: doc, predict_class (2 classes P, N)
    def get_confusion(self):
        self.df_confusion = pd.crosstab(self.df["actual"], self.df["predicted"], rownames=["Actual"], colnames=["Predicted"], margins=True)
        # print(self.df_confusion)
    
    def get_accuracy(self):
        return (self.df_confusion.loc["P", "P"] + self.df_confusion.loc["N", "N"])/self.df_confusion.loc["All", "All"]
    
    def get_precision(self):
        return self.df_confusion.loc["P", "P"]/(self.df_confusion.loc["P", "P"] + self.df_confusion.loc["N", "P"])
    
    def get_recall(self):
        return self.df_confusion.loc["P", "P"]/(self.df_confusion.loc["P", "P"] + self.df_confusion.loc["P", "N"])
    
    def plot_confusion_matrix(self, title='confusion_matrix', cmap=plt.cm.get_cmap("viridis_r")):
        plt.matshow(self.df_confusion, cmap=cmap) # imshow
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.df_confusion.columns))
        plt.xticks(tick_marks, self.df_confusion.columns, rotation=45)
        plt.yticks(tick_marks, self.df_confusion.index)
        #plt.tight_layout()
        plt.ylabel(self.df_confusion.index.name)
        plt.xlabel(self.df_confusion.columns.name)
        plt.savefig(title + ".png")

def plot_graph(x, y, filename):
    plt.figure(figsize=(10, 6))
    plt.semilogx(x, y, marker='o', linestyle='-', color='b')
    plt.title("Model Accuracy vs. Alpha (Laplace Smoothing)")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Accuracy on Test Set")
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename + ".png")
    plt.show()                