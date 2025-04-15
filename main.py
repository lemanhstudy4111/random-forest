import numpy as np
from DecisionTree import RandomForest
from Utils import Performance, StratifiedValidation
from itertools import chain


def main ():
    filename = "/home/lemin/random-forest/data/loan.csv"
    ds = np.genfromtxt(filename, delimiter=',', names=True, dtype=None)
    ntree_values = [1, 5, 10, 20, 30, 40, 50]
    stratValidation = StratifiedValidation(dataset=ds, k=5)
    kfolds = stratValidation.kfold_split()
    for dataset in kfolds:
        validation_set = dataset
        kfolds.pop(0)
        print(len(kfolds))
        rf_model = RandomForest(dataset=list(chain.from_iterable(kfolds)), ntree=50)
        rf_model.train_random_forest()
        all_preds = rf_model.predict(validation_set)
        performance_validation = Performance(y=validation_set["label"], preds=all_preds)
        print("accuracy ", performance_validation.get_accuracy())
        print("precision ", performance_validation.get_precision())
        print("recall ", performance_validation.get_recall())
        print("f1 ", performance_validation.get_f1())
        kfolds.append(validation_set)

main()