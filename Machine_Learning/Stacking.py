import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from variables import *
from GenericClassifier import BlirModel

# Uncomment next line to create only one classifier

estimators = [
    ("nb", MultinomialNB(alpha=0.01)),
    (
        "dt",
        DecisionTreeClassifier(
            class_weight="balanced", random_state=0, min_samples_leaf=5
        ),
    ),
    (
        "svm",
        SVC(
            class_weight="balanced", kernel="rbf", gamma="scale", random_state=0, C=0.1
        ),
    ),
    (
        "rf",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=0,
            bootstrap=False,
            max_depth=20,
            min_samples_leaf=2,
            n_estimators=100,
        ),
    ),
    (
        "lr",
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            random_state=0
            #    ),
        ),
    ),
    ("nn", MLPClassifier(solver="lbfgs", random_state=0)),
]

#  A    ngram: 1    df: 1    Tf-idf: True
# Features: (3016, 18028)
# Precision        Recall          F-score         ROC AUC
# 0.7318681318681319   0.3770833333333333      0.48085267516617636     0.9523076923076923

# final_estimator = LogisticRegression()
# final_estimator = MultinomialNB(alpha=0.01)
# final_estimator = RandomForestClassifier(class_weight = 'balanced',\
#                                         random_state = 0, bootstrap = False,
#                                                                          max_depth = 20, min_samples_leaf = 2, n_estimators = 100)
# final_estimator = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0)
# bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
final_estimator = DecisionTreeClassifier(
    class_weight="balanced", random_state=0, min_samples_leaf=5
)
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# clf = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0)
model = BlirModel(clf, "Stacking")
# model.single_run(DIET, ngram=3, df=23, tfidf=False, class_type="A")
# model.single_run(DIET, ngram=1, df=21, tfidf=True, class_type="A")
model.multiple_runs_write_files(DIET, ["A"])

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
