from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from variables import *
from GenericClassifier import BlirModel
import numpy as np

# Uncomment next line to create only one classifier

# KNeighborsClassifier(),
bagging = BaggingClassifier(MultinomialNB(alpha=0.01))
bagging = BaggingClassifier(MLPClassifier(solver="lbfgs", random_state=0))
bagging = BaggingClassifier(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=0,
        bootstrap=False,
        max_depth=20,
        min_samples_leaf=2,
        n_estimators=100,
    )
)
# bagging = BaggingClassifier(
#    LogisticRegression(
#        C=0.1, solver="liblinear", class_weight="balanced", random_state=0
#    ),
# )
# bagging = BaggingClassifier(
#    DecisionTreeClassifier(class_weight="balanced", random_state=0, min_samples_leaf=5)
# )

# clf = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0)
model = BlirModel(bagging, "Bagging")
# model.single_run(DIET, ngram=1, df=1, tfidf=True, class_type="A")

model.multiple_runs_write_files(DIET, ["A"])

max_samples = [int(x) for x in np.linspace(1, 10, num=10)]
# max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

# param_grid = {"min_samples_leaf": min_samples_leaf, "max_depth": max_depth}

# model.single_run_grid(
#    DIET, ngram=1, df=1, tfidf=True, class_type="T", param_grid=param_grid
# )
