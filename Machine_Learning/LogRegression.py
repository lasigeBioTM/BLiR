from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *



def CrossVal_LogReg(X_matrix, y_labels, C):
    """
    Logistic Regression using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
              C: Inverse of regularization strength.
    Ensures: Precision, Recall and F-score for class 1 (relevant).
    """

    log = linear_model.LogisticRegression(C = C, solver = 'liblinear', class_weight = 'balanced', random_state = 0)
    scoring = ['precision', 'recall', 'f1']
    scores = cross_validate(log, X_matrix, y_labels, cv = 10, scoring = scoring)

    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    fscore = scores['test_f1'].mean()
    
    print('Precision \t\t Recall  \t\t F-score  \t\t C')
    print(precision, '\t', recall, '\t', fscore, '\t', C, '\n')
    
    return precision, recall, fscore



def grid_search_LogReg(X_matrix, y_labels):
    """
    Searches for the best value of C to use as parameter in the Logistic Regression algorithm

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
    Ensures: Precision, Recall, F-score and best C
    """   

    clf = linear_model.LogisticRegression(solver = 'liblinear', class_weight = 'balanced', random_state = 0)

    param_grid = {'C':[0.001, 0.01, 0.1, 1.0, 10.0]}
    
    scoring = ['precision', 'recall','f1']
    grid = GridSearchCV(clf, param_grid, iid = False, cv = 10, scoring = scoring, refit = 'f1')

    best_model = grid.fit(X_matrix, y_labels)
    
    C = best_model.best_estimator_.get_params()['C']
    precision = best_model.cv_results_['mean_test_precision'][best_model.best_index_]
    recall = best_model.cv_results_['mean_test_recall'][best_model.best_index_]
    fscore = best_model.cv_results_['mean_test_f1'][best_model.best_index_]

    print('Precision \t\t Recall  \t\t F-score  \t\t C')
    print(precision,'\t', recall,'\t', fscore, '\t', C,'\n')

    return precision, recall, fscore, C



def WriteFiles(file, tfidf, class_type, C):
    """
    Write files with the results of the LogReg, for different combinations of parameters
        - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
        - different values of min_df to build the matrices
        - different ngrams(used to create the features)
    """

    try:
        out_file = open(LG_PATH/('LogReg2_'+ file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(LG_PATH/('LogReg_'+ file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score \t\t\t\t C\n')
        
        for df in range(1+ngram, 21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F = CrossVal_LogReg(X_matrix, y_labels, C)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) + '\t\t' + str(F) + '\t\t' + str(C) + '\n')
            
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()



def Get_classification_report(X_matrix, y_labels, C):
    """
    Get the model's predictions
    
    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
              C: Inverse of regularization strength.
    Ensures: all_pred: list with prediction, each position corresponds to an article,
             for example, the first element of the list corresponds to the first article
             in the files in the ML_data folder.
    """

    k_folds = model_selection.StratifiedKFold(n_splits = 10)

    #metrics
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    all_pred_dic = {}
        
    for train_index, test_index in k_folds.split(X_matrix, y_labels):
        # split the data into the training and testing set
        X_train, X_test = X_matrix[train_index], X_matrix[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]

        clf = linear_model.LogisticRegression(C=C, solver = 'liblinear', class_weight = 'balanced')
        clf.fit(X_train, y_train)
        y_predictions = clf.predict(X_test)

        # get predictions
        for index in range(len(test_index)):
            all_pred_dic[test_index[index]] = y_predictions[index]

        # get scores
        recall_scores.append(metrics.recall_score(y_test, y_predictions))
        precision_scores.append(metrics.precision_score(y_test, y_predictions))
        f1_scores.append(metrics.f1_score(y_test, y_predictions))

    # map predictions to the correct position of the articles in the list
    all_pred = [all_pred_dic[j] for j in range(len(all_pred_dic))]

    print('Number of Positives', all_pred.count(1), '\nNumber of Negatives', all_pred.count(0))
    
    recall = sum(recall_scores)/float(len(recall_scores))
    precision = sum(precision_scores)/float(len(precision_scores))
    f1 = sum(f1_scores)/float(len(f1_scores))

    print('Precision \t\t Recall  \t\t F-score  \t\t C')
    print(precision, '\t', recall, '\t', f1, '\t', C, '\n')
    
    return all_pred





""" ----------------------------------- Single Run ----------------------------------- """

def single_run_CV(file, ngram, df, tfidf, class_type, C):
    """
    Run CrossVal_LogReg(X_matrix, y_labels, C)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_LogReg(X_matrix, y_labels, C)
    
# Uncomment next line to create only one classifier
#single_run_CV(DIET, ngram = 2, df = 14, tfidf = True, class_type = 'A', C = 10.0)



def single_run_grid(file, ngram, df, tfidf, class_type):
    """
    Run grid_search_LogReg(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    grid_search_LogReg(X_matrix, y_labels)

# Uncomment next line to run a grid search
#single_run_grid(DIET, ngram = 1, df = 1, tfidf = True, class_type = 'T')



def single_run_CR(file, ngram, df, tfidf, class_type, C):
    """
    Run Get_classification_report(X_matrix, y_labels, C)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    all_pred = Get_classification_report(X_matrix, y_labels, C)
    return all_pred

# Uncomment the next lines to get the label predictions from a single classifier
#all_pred = single_run_CR(DIET, ngram = 1, df = 1, tfidf = False, class_type = 'T', C = 0.1) 
#print(all_pred)




""" ------------------------- Multiple Runs and write files -------------------------- """

def multiple_runs_write_files(file):
    """
    Run WriteFiles(file, tfidf, class_type, C)
    """
    
    # Run for Titles   
    WriteFiles(file, tfidf = True, class_type = 'T', C = 1.0)
    WriteFiles(file, tfidf = False, class_type = 'T', C = 0.1)
    
    # Run for Abstracts
    WriteFiles(file, tfidf = True, class_type = 'A', C = 10.0)
    WriteFiles(file, tfidf = False, class_type = 'A', C = 0.1)
    
    # Run for Metadata
    WriteFiles(file, tfidf = True, class_type = 'M', C = 0.1)
    WriteFiles(file, tfidf = False, class_type = 'M', C = 0.1)
    
    # Run for Titles and Meta
    WriteFiles(file, tfidf = True, class_type = 'TM', C = 1.0)
    WriteFiles(file, tfidf = False, class_type = 'TM', C = 0.1)


# Uncomment next line to get the files in the Results/LogReg folder
#multiple_runs_write_files(DIET)

