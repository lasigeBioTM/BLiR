from sklearn.ensemble import RandomForestClassifier
import numpy as np
from variables import *
from GenericClassifier import BlirModel


""" ----------------------------- Single Run ----------------------------- """


# Uncomment next line to create only one classifier
clf = RandomForestClassifier(
    class_weight="balanced",
    random_state=0,
    bootstrap=False,
    max_depth=20,
    min_samples_leaf=2,
    n_estimators=100,
)
model = BlirModel(clf, "RandomForest")
model.single_run(DIET, ngram=3, df=4, tfidf=False, class_type="AMT")
# all_pred, TN, FP, FN, TP = model.single_run_classification_report(
#    DIET, ngram=3, df=4, tfidf=False, class_type="A"
# )
# print(len(all_pred))
# print("TN:\n", len(TN), "\nFP:\n", len(FP), "\nFN:\n", len(FN), "\nTP:\n", len(TP))

""" ------------------- Multiple Runs and write files -------------------- """


# Uncomment next line to get the files in the Results/RandomForest folder
# model.multiple_runs_write_files(DIET)

# Uncomment next line to run a grid search
# clf = RandomForestClassifier(
#    class_weight="balanced", n_estimators=100, random_state=0, bootstrap=False
# )
# model = BlirModel(clf, "RandomForest")
# min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=10)]
# max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

# param_grid = {"min_samples_leaf": min_samples_leaf, "max_depth": max_depth}

# model.single_run_grid(
#    DIET, ngram=1, df=1, tfidf=True, class_type="T", param_grid=param_grid
# )
