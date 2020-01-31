from variables import *
from GenericClassifier import BlirModel
from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *

log = linear_model.LogisticRegression(
    C=10, solver="liblinear", class_weight="balanced", random_state=0
)
param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
clf = linear_model.LogisticRegression(
    solver="liblinear", class_weight="balanced", random_state=0, C=0.1
)


model = BlirModel(clf, "LR")
model.single_run(DIET, ngram=1, df=14, tfidf=False, class_type="M")
model.single_run(DIET, ngram=1, df=14, tfidf=False, class_type="MT")
model.single_run(ALL, ngram=1, df=14, tfidf=False, class_type="M")
model.single_run(ALL, ngram=1, df=14, tfidf=False, class_type="MT")
# model.single_run_classification_report(DIET, ngram=1, df=14, tfidf=True, class_type="A")
