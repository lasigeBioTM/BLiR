from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *



def CrossVal_RandomForest(X_matrix, y_labels):
    """
    Random Forest using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels.
    Ensures: Precision, Recall and F-score for class 1 (relevant).
    """

    clf = RandomForestClassifier(class_weight = 'balanced',\
                                 random_state = 0, bootstrap = False,
                                 max_depth = 20, min_samples_leaf = 2, n_estimators = 100)
    
    scoring = ['precision', 'recall','f1']
    scores = cross_validate(clf, X_matrix, y_labels, cv = 10, scoring = scoring)
    
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    fscore = scores['test_f1'].mean()
    
    print('Precision \t\t Recall  \t\t F-score')
    print(precision,'\t', recall,'\t', fscore, '\n')

    return precision, recall, fscore




def grid_search_RF(X_matrix, y_labels):
    """
    Searches for the best values to use as parameter in the Random Forest algorithm

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
    Assures: Precision, Recall, F-score and best min_samples_leaf and max_depth
    """

    clf = RandomForestClassifier(class_weight = 'balanced', n_estimators = 100,\
                                 random_state = 0, bootstrap = False)

    min_samples_leaf = [int(x) for x in np.linspace(1, 10, num = 10)]
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

    param_grid = {'min_samples_leaf':min_samples_leaf, 'max_depth':max_depth}
    
    scoring = ['precision', 'recall','f1']
    grid = GridSearchCV(clf, param_grid, cv = 10, scoring = scoring, \
                        refit = 'f1', iid = False)

    best_model = grid.fit(X_matrix, y_labels)

    min_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
    max_depth = best_model.best_estimator_.get_params()['max_depth']
    precision = best_model.cv_results_['mean_test_precision'][best_model.best_index_]
    recall = best_model.cv_results_['mean_test_recall'][best_model.best_index_]
    fscore = best_model.cv_results_['mean_test_f1'][best_model.best_index_]

    print('Precision \t Recall  \t F-score  \t max_depth  \t min_samples_leaf')
    print(precision,'\t\t', recall,'\t\t', fscore,'\t\t', max_depth,'\t\t', min_samples_leaf, '\n')

    return precision, recall, fscore, max_depth, min_samples_leaf




def WriteFiles(file, tfidf, class_type):
    """
    Write files with the results of the LogReg, for different combinations of parameters
        - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
        - different values of min_df to build the matrices
        - different ngrams(used to create the features)
    """

    try:
        out_file = open(RF_PATH/('RandomForest_'+ file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(RF_PATH/('RandomForest_'+ file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score\n')
        
        for df in range(1+ngram, 21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F  = CrossVal_RandomForest(X_matrix, y_labels)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) +'\t\t' + str(F) + '\n')
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()
    



""" ----------------------------- Single Run ----------------------------- """

def single_run_CV(file, ngram, df, tfidf, class_type):
    """
    CrossVal_RandomForest(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_RandomForest(X_matrix, y_labels)

# Uncomment next line to create only one classifier
#single_run_CV(file, ngram = 3, df = 4, tfidf = False, class_type = 'A')



def single_run_grid(file, ngram, df, tfidf, class_type):
    """
    Run grid_search_RF(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    grid_search_RF(X_matrix, y_labels)

# Uncomment next line to run a grid search
#single_run_grid(file, ngram = 1, df = 1, tfidf = True, class_type = 'T')




""" ------------------- Multiple Runs and write files -------------------- """

def multiple_runs_write_files(file):
    """
    Run WriteFiles(file, tfidf, class_type)
    """

    ## Run for Titles   
    WriteFiles(file, tfidf = True, class_type = 'T')
    WriteFiles(file, tfidf = False, class_type = 'T')

    ## Run for Abstracts
    WriteFiles(file, tfidf = True, class_type = 'A')
    WriteFiles(file, tfidf = False, class_type = 'A')

    ## Run for Metadata
    WriteFiles(file, tfidf = True, class_type = 'M')
    WriteFiles(file, tfidf = False, class_type = 'M')

    ## Run for Titles and Meta
    WriteFiles(file, tfidf = True, class_type = 'TM')
    WriteFiles(file, tfidf = False, class_type = 'TM')


# Uncomment next line to get the files in the Results/RandomForest folder
#multiple_runs_write_files(file)

