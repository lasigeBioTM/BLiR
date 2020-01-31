import numpy as np
from sklearn import tree
from variables import *
from GenericClassifier import BlirModel


clf = tree.DecisionTreeClassifier(
    class_weight="balanced", random_state=0, min_samples_leaf=5
)

clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0)

min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=10)]

# param_grid = {"min_samples_leaf": min_samples_leaf}


""" ----------------------------- Single Run ----------------------------- """
model = BlirModel(clf, "DecisionTree")
model.single_run(DIET, ngram=3, df=4, tfidf=False, class_type="AT")


def single_run_grid(file, ngram, df, tfidf, class_type):
    """
    Run grid_search_DT(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    grid_search_DT(X_matrix, y_labels)


# Uncomment next line to run a grid search
# single_run_grid(DIET, ngram = 1, df = 1, tfidf = False, class_type = 'T')


""" ------------------- Multiple Runs and write files -------------------- """


def multiple_runs_write_files(file):
    """
    Run WriteFiles(file, tfidf, class_type)
    """
    ## Run for Titles
    WriteFiles(file, tfidf=True, class_type="T")
    WriteFiles(file, tfidf=False, class_type="T")

    ## Run for Abstracts
    WriteFiles(file, tfidf=True, class_type="A")
    WriteFiles(file, tfidf=False, class_type="A")

    ## Run for Metadata
    WriteFiles(file, tfidf=True, class_type="M")
    WriteFiles(file, tfidf=False, class_type="M")

    ## Run for Titles and Meta
    WriteFiles(file, tfidf=True, class_type="TM")
    WriteFiles(file, tfidf=False, class_type="TM")


# Uncomment next line to get the files in the Results/DecisionTree folder
# multiple_runs_write_files(DIET)
