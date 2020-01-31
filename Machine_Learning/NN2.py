from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *

clf = MLPClassifier(solver="lbfgs", random_state=0)


# Uncomment next line to create only one classifier
# single_run_CV(DIET, ngram = 3, df = 16, tfidf = False, class_type = 'A')


# Uncomment next line to get the files in the Results/NeuralNetwork folder
# multiple_runs_write_files(DIET)
