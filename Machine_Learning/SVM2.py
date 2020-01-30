from sklearn.svm import SVC

from variables import *
from GenericClassifier import BlirModel

# Uncomment next line to create only one classifier
clf = SVC(class_weight="balanced", kernel="rbf", gamma="scale", random_state=0, C=0.1)
model = BlirModel(clf, "SVM")
# model.single_run(DIET, ngram=1, df=21, tfidf=False, class_type="A")
# model.single_run_classification_report(DIET, ngram=1, df=21, tfidf=True, class_type="A")
# model.multiple_runs_write_files(DIET)
# Uncomment the next lines to get the label predictions from a single classifier
all_pred, TN, FP, FN, TP = model.single_run_classification_report(
    DIET, ngram=1, df=21, tfidf=False, class_type="T"
)

positives = TP + FP

for i in positives:
    print(i)

# print(all_pred)
# print('TN:\n', TN, '\nFP:\n', FP, '\nFN:\n', FN, '\nTP:\n', TP)


""" ------------------- Multiple Runs and write files -------------------- """


# Uncomment next line to get the files in the Results/SVM folder
# multiple_runs_write_files(DIET)
