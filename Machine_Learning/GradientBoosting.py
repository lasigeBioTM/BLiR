import numpy as np


from sklearn.ensemble import GradientBoostingClassifier
from variables import *
from GenericClassifier import BlirModel

# Uncomment next line to create only one classifier


# bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
)

# clf = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0)
model = BlirModel(clf, "GradientBoosting")
model.single_run(DIET, ngram=1, df=1, tfidf=True, class_type="A")

# model.multiple_runs_write_files(DIET)

n_estimators = range(50, 1050, 200)
min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=10)]
# max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]


param_grid = {
    "min_samples_leaf": min_samples_leaf,
    "n_estimators": n_estimators,
    # "max_depth": max_depth,
}

# model.single_run_grid(
#    DIET, ngram=1, df=1, tfidf=True, class_type="T", param_grid=param_grid
# )
