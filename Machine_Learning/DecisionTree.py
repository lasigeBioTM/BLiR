from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

from data_preprocessing import *



def CrossVal_DecisionTree(X_matrix, y_labels):
    """
    Decision Tree using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels.
    Ensures: Precision, Recall and F-score for class 1 (relevant).
    """

    clf = tree.DecisionTreeClassifier(class_weight = 'balanced',\
                                      random_state = 0, min_samples_leaf = 5)

    scoring = ['precision', 'recall','f1']
    scores = cross_validate(clf, X_matrix, y_labels, cv = 10, scoring = scoring)

    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    fscore = scores['test_f1'].mean()
    
    print('Precision \t\t Recall  \t\t F-score')
    print(precision, '\t', recall, '\t', fscore,'\n')

    return precision, recall, fscore




def grid_search_DT(X_matrix, y_labels):
    """
    Searches for the best values of min_samples_leaf to use in the Decision Tree algorithm

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
    Assures: Precision, Recall, F-score and best min_samples_leaf
    """

    clf = tree.DecisionTreeClassifier(class_weight = 'balanced', \
                                      random_state = 0)
    
    min_samples_leaf = [int(x) for x in np.linspace(1, 10, num = 10)]

    param_grid = {'min_samples_leaf':min_samples_leaf}
    
    scoring = ['precision', 'recall', 'f1']
    grid = GridSearchCV(clf, param_grid, cv = 10, scoring = scoring, \
                        refit = 'f1')

    best_model = grid.fit(X_matrix, y_labels)

    min_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
    precision = best_model.cv_results_['mean_test_precision'][best_model.best_index_]
    recall = best_model.cv_results_['mean_test_recall'][best_model.best_index_]
    fscore = best_model.cv_results_['mean_test_f1'][best_model.best_index_]

    print('Precision \t\t Recall  \t\t F-score  \t\t min_samples_leaf')
    print(precision,'\t', recall,'\t', fscore,'\t', min_samples_leaf, '\n')

    return precision, recall, fscore, min_samples_leaf




def WriteFiles(file, tfidf, class_type):
    """
    Write files with the results of the DecTree, for different combinations of parameters
        - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
        - different values of min_df to build the matrices
        - different ngrams(used to create the features)
    """

    try:
        out_file = open(DT_PATH/('DecisionTree_' + file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(DT_PATH/('DecisionTree_' + file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t Recall \t\t\t F-score\n')
        
        for df in range(1+ngram, 21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F = CrossVal_DecisionTree(X_matrix, y_labels)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) +'\t\t' + str(F) + '\n')
                           
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()





""" ----------------------------- Single Run ----------------------------- """

def single_run_CV(file, ngram, df, tfidf, class_type):
    """
    Run CrossVal_DecisionTree(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_DecisionTree(X_matrix, y_labels)

# Uncomment next line to create only one classifier
#single_run_CV(DIET, ngram = 3, df = 4, tfidf = False, class_type = 'A')



def single_run_grid(file, ngram, df, tfidf, class_type):
    """
    Run grid_search_DT(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    grid_search_DT(X_matrix, y_labels)

# Uncomment next line to run a grid search
#single_run_grid(DIET, ngram = 1, df = 1, tfidf = False, class_type = 'T')





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


# Uncomment next line to get the files in the Results/DecisionTree folder
#multiple_runs_write_files(DIET)


