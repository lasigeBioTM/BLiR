# generic_classifier_class
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection, metrics

from data_preprocessing import *


class BlirModel:
    def __init__(self, clf, clf_name):
        self.clf = clf
        self.clf_name = clf_name
        self.n_jobs = 10

    def CrossVal(self, X_matrix, y_labels):
        """
        Suport Vector Machine using Cross Validation

        Requires: X_matrix: a doc-term count or tfidf matrix;
                  y_labels: a matrix with labels.
        Ensures: Precision, Recall and F-score for class 1 (relevant).
        """

        # clf = SVC(class_weight = 'balanced', kernel = 'rbf', gamma='scale', random_state = 0)

        scoring = ["precision", "recall", "f1", "roc_auc"]
        scores = cross_validate(
            self.clf, X_matrix, y_labels, cv=10, scoring=scoring, n_jobs=self.n_jobs
        )
        # print(scores)

        precision = scores["test_precision"].mean()
        recall = scores["test_recall"].mean()
        fscore = scores["test_f1"].mean()
        roc_auc = scores["test_roc_auc"].mean()

        print("Precision \t\t Recall  \t\t F-score \t\t ROC AUC")
        print(precision, "\t", recall, "\t", fscore, "\t", roc_auc, "\n")

        return precision, recall, fscore

    def grid_search(self, X_matrix, y_labels, param_grid):
        """
        Searches for the best values to use as parameter in the Random Forest algorithm

        Requires: X_matrix: a doc-term count or tfidf matrix;
                  y_labels: a matrix with labels;
        Assures: Precision, Recall, F-score and best min_samples_leaf and max_depth
        """

        # clf = RandomForestClassifier(
        #    class_weight="balanced", n_estimators=100, random_state=0, bootstrap=False
        # )

        scoring = ["precision", "recall", "f1"]
        grid = GridSearchCV(
            self.clf,
            param_grid,
            cv=10,
            scoring=scoring,
            refit="f1",
            iid=False,
            n_jobs=self.n_jobs,
            verbose=1,
        )

        best_model = grid.fit(X_matrix, y_labels)

        # min_samples_leaf = best_model.best_estimator_.get_params()["min_samples_leaf"]
        # max_depth = best_model.best_estimator_.get_params()["max_depth"]
        best_params = best_model.best_estimator_.get_params()
        precision = best_model.cv_results_["mean_test_precision"][
            best_model.best_index_
        ]
        recall = best_model.cv_results_["mean_test_recall"][best_model.best_index_]
        fscore = best_model.cv_results_["mean_test_f1"][best_model.best_index_]

        print("Precision \t Recall  \t F-score \t" + "\t".join(param_grid.keys()))
        print(
            precision,
            "\t\t",
            recall,
            "\t\t",
            fscore,
            "\t\t",
            "\t\t".join([str(best_params[x]) for x in param_grid.keys()]),
        )

        return precision, recall, fscore, best_params

    def writeFiles(self, file, tfidf, class_type):
        """
        Write files with the results of the LogReg, for different combinations of parameters
            - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
            - different values of min_df to build the matrices
            - different ngrams(used to create the features)
        """

        try:
            out_file = open(
                RESULTS_PATH
                / self.clf_name
                / (self.clf_name + "_" + file + "_" + class_type + ".txt"),
                "a",
            )
        except:
            out_file = open(
                RESULTS_PATH
                / self.clf_name
                / (self.clf_name + "_" + file + "_" + class_type + ".txt"),
                "w",
            )

        out_file.write(
            "\n------------------------------- TF-IDF "
            + str(tfidf).upper()
            + " -------------------------------\n"
        )

        for ngram in range(1, 4):
            out_file.write(
                "n-grams(1-"
                + str(ngram)
                + ")\n"
                + "df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score\n"
            )

            for df in range(1 + ngram, 21 + ngram):
                X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
                P, R, F = self.CrossVal(X_matrix, y_labels)
                out_file.write(
                    str(df) + "\t" + str(P) + "\t\t" + str(R) + "\t\t" + str(F) + "\n"
                )

            out_file.write("\n")

        out_file.write("\n\n")
        out_file.close()

    def Get_classification_report(self, X_matrix, y_labels):
        """
        Get the model's predictions
        
        Requires: X_matrix: a doc-term count or tfidf matrix;
                  y_labels: a matrix with labels.
        Ensures: all_pred: list with prediction, each position corresponds to an article,
                 for example, the first element of the list corresponds to the first article
                 in the files in the ML_data folder.
                 TN (True Negatives), FP (False Positives),
                 FN (False Negatives), TP (True Positives).
        """

        k_folds = model_selection.StratifiedKFold(n_splits=10)

        scores = []
        all_pred_dic = {}
        TN, FP, FN, TP = [], [], [], []

        for train_index, test_index in k_folds.split(X_matrix, y_labels):
            # split the data into the training and testing set
            X_train, X_test = X_matrix[train_index], X_matrix[test_index]
            y_train, y_test = y_labels[train_index], y_labels[test_index]

            # clf = SVC(class_weight = 'balanced', gamma='scale', kernel = 'rbf')
            self.clf.fit(X_train, y_train)
            y_predictions = self.clf.predict(X_test)

            for index in range(len(test_index)):
                # get predictions
                all_pred_dic[test_index[index]] = y_predictions[index]

                # get FN, FP, TN, TP
                if y_test[index] != y_predictions[index]:
                    if y_test[index] == 1:
                        FN.append(test_index[index])
                    else:
                        FP.append(test_index[index])
                else:
                    if y_test[index] == 1:
                        TP.append(test_index[index])
                    else:
                        TN.append(test_index[index])

            scores.append(metrics.recall_score(y_test, y_predictions))

        # map predictions to the correct position of the articles in the list
        all_pred = [all_pred_dic[j] for j in range(len(all_pred_dic))]

        print("TN:", len(TN), "\nFP:", len(FP), "\nFN:", len(FN), "\nTP:", len(TP))
        print(
            "Number of Positives",
            all_pred.count(1),
            "\nNumber of Negatives",
            all_pred.count(0),
        )

        recall = sum(scores) / float(len(scores))
        print("\nRecall:", recall)

        return all_pred, TN, FP, FN, TP

    def single_run(self, file, ngram, df, tfidf, class_type):
        """
        Run CrossVal_SVC(X_matrix, y_labels)
        """
        X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
        self.CrossVal(X_matrix, y_labels)

    def single_run_classification_report(self, file, ngram, df, tfidf, class_type):
        """
        Run Get_classification_report(X_matrix, y_labels)
        """
        X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
        all_pred, TN, FP, FN, TP = self.Get_classification_report(X_matrix, y_labels)
        return all_pred, TN, FP, FN, TP

    def single_run_grid(self, file, ngram, df, tfidf, class_type, param_grid):
        """
        Run grid_search_RF(X_matrix, y_labels)
        """
        X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
        self.grid_search(X_matrix, y_labels, param_grid)

    def multiple_runs_write_files(self, file):
        """
        Run WriteFiles(file, tfidf, class_type)
        """

        print("Run for Titles")
        self.writeFiles(file, tfidf=True, class_type="T")
        self.writeFiles(file, tfidf=False, class_type="T")

        print("Run for Abstracts")
        self.writeFiles(file, tfidf=True, class_type="A")
        self.writeFiles(file, tfidf=False, class_type="A")

        print("Run for Metadata")
        self.writeFiles(file, tfidf=True, class_type="M")
        self.writeFiles(file, tfidf=False, class_type="M")

        print("Run for Titles and Meta")
        self.writeFiles(file, tfidf=True, class_type="TM")
        self.writeFiles(file, tfidf=False, class_type="TM")
