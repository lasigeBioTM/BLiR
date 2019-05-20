from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection, metrics

from data_preprocessing import *



def CrossVal_SVC(X_matrix, y_labels):
    """
    Suport Vector Machine using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels.
    Ensures: Precision, Recall and F-score for class 1 (relevant).
    """

    clf = SVC(class_weight = 'balanced', kernel = 'rbf', gamma='scale', random_state = 0)
    
    scoring = ['precision', 'recall','f1']
    scores = cross_validate(clf, X_matrix, y_labels, cv = 10, scoring = scoring)
    
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    fscore = scores['test_f1'].mean()
    
    print('Precision \t\t Recall  \t\t F-score')
    print(precision,'\t', recall,'\t', fscore, '\n')

    return precision, recall, fscore



def WriteFiles(file, tfidf, class_type):
    """
    Write files with the results of the LogReg, for different combinations of parameters
        - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
        - different values of min_df to build the matrices
        - different ngrams(used to create the features)
    """

    try:
        out_file = open(SVM_PATH/('SVM_'+ file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(SVM_PATH/('SVM_'+ file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score\n')
        
        for df in range(1+ngram, 21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F = CrossVal_SVC(X_matrix, y_labels)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) + '\t\t' + str(F) + '\n')
            
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()




def Get_classification_report(X_matrix, y_labels):
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
    
    k_folds = model_selection.StratifiedKFold(n_splits = 10)

    scores = []
    all_pred_dic = {}
    TN, FP, FN, TP = [],[],[],[]
        
    for train_index, test_index in k_folds.split(X_matrix, y_labels):
        # split the data into the training and testing set
        X_train, X_test = X_matrix[train_index], X_matrix[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]

        clf = SVC(class_weight = 'balanced', gamma='scale', kernel = 'rbf')
        clf.fit(X_train, y_train)
        y_predictions = clf.predict(X_test)
    
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

    print('TN:', len(TN), '\nFP:', len(FP), '\nFN:', len(FN), '\nTP:', len(TP))
    print('Number of Positives', all_pred.count(1), '\nNumber of Negatives', all_pred.count(0))
    
    recall = sum(scores)/float(len(scores))
    print('\nRecall:', recall)
    
    return all_pred, TN, FP, FN, TP





""" ----------------------------- Single Run ----------------------------- """

def single_run_SVC(file, ngram, df, tfidf, class_type):
    """
    Run CrossVal_SVC(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_SVC(X_matrix, y_labels)

# Uncomment next line to create only one classifier  
#single_run_SVC(DIET, ngram = 1, df = 21, tfidf = True, class_type = 'A')



def single_run_classification_report(file, ngram, df, tfidf, class_type):
    """
    Run Get_classification_report(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    all_pred, TN, FP, FN, TP = Get_classification_report(X_matrix, y_labels)
    return all_pred, TN, FP, FN, TP

# Uncomment the next lines to get the label predictions from a single classifier
#all_pred, TN, FP, FN, TP = single_run_classification_report(DIET, ngram = 1, df = 1, tfidf = True, class_type = 'T')
#print(all_pred)
#print('TN:\n', TN, '\nFP:\n', FP, '\nFN:\n', FN, '\nTP:\n', TP)




""" ------------------- Multiple Runs and write files -------------------- """

def multiple_runs_write_files(file):
    """
    Run WriteFiles(file, tfidf, class_type)
    """
    
    # Run for Titles   
    WriteFiles(file, tfidf = True, class_type = 'T')
    WriteFiles(file, tfidf = False, class_type = 'T')

    # Run for Abstracts
    WriteFiles(file, tfidf = True, class_type = 'A')
    WriteFiles(file, tfidf = False, class_type = 'A')

    # Run for Metadata
    WriteFiles(file, tfidf = True, class_type = 'M')
    WriteFiles(file, tfidf = False, class_type = 'M')

    # Run for Titles and Meta
    WriteFiles(file, tfidf = True, class_type = 'TM')
    WriteFiles(file, tfidf = False, class_type = 'TM')


# Uncomment next line to get the files in the Results/SVM folder
#multiple_runs_write_files(DIET)

