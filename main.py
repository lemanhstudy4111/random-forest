import numpy as np
from DecisionTree import RandomForest
from Utils import Performance, StratifiedValidation
from itertools import chain


def main ():
    filename = "/home/lemin/random-forest/data/wdbc.csv"
    ds = np.genfromtxt(filename, delimiter=',', names=True)
    stratValidation = StratifiedValidation(dataset=ds, k=5)
    kfolds = stratValidation.kfold_split()
    for dataset in kfolds:
        validation_set = dataset
        kfolds.pop(0)
        rf_model = RandomForest(dataset=list(chain.from_iterable(kfolds)), ntree=1)
        rf_model.train_random_forest()
        all_preds = rf_model.predict(validation_set)
        performance_validation = Performance(y=validation_set[:, "label"], preds=all_preds)
        print("accuracy ", performance_validation.get_accuracy())
        print("precision ", performance_validation.get_precision())
        print("recall ", performance_validation.get_recall())
        print("f1 ", performance_validation.get_f1())
        kfolds.append(validation_set)

main()