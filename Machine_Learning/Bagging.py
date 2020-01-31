from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from variables import *
from GenericClassifier import BlirModel
import numpy as np

# Uncomment next line to create only one classifier

# KNeighborsClassifier(),
bagging = BaggingClassifier(
    KNeighborsClassifier(), n_estimators=10, max_samples=1.0, max_features=1.0
)

bagging = BaggingClassifier(
    LogisticRegression(), n_estimators=100, max_samples=1.0, max_features=1.0
)


# clf = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0)
model = BlirModel(bagging, "Bagging")
model.single_run(DIET, ngram=1, df=1, tfidf=True, class_type="A")

# model.multiple_runs_write_files(DIET)

max_samples = [int(x) for x in np.linspace(1, 10, num=10)]
# max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

# param_grid = {"min_samples_leaf": min_samples_leaf, "max_depth": max_depth}

# model.single_run_grid(
#    DIET, ngram=1, df=1, tfidf=True, class_type="T", param_grid=param_grid
# )
