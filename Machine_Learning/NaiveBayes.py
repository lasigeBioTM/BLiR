from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *



def CrossVal_NaiveBayes(X_matrix, y_labels):
    """
    Naive Bayes using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix
              y_labels: a matrix with labels
              alpha: "Additive (Laplace/Lidstone) smoothing parameter." 
    Ensures: Precision, Recall and F-score for class 1 (relevant)
    """
    
    gnb = MultinomialNB(alpha = 0.01)

    scoring = ['precision', 'recall', 'f1']
    scores = cross_validate(gnb, X_matrix, y_labels, cv = 10, scoring = scoring)
    
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    fscore = scores['test_f1'].mean()
    
    print('Precision \t\t Recall  \t\t F-score')
    print(precision,'\t', recall,'\t', fscore, '\n')

    return precision, recall, fscore




def grid_search(X_matrix, y_labels):
    """
    Searches for the best value of alpha to use as parameter in the Naive Bayes algorithm

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels;
    Assures: Precision, Recall, F-score and best alpha
    """
   
    clf = MultinomialNB()

    param_grid = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}

    scoring = ['precision', 'recall','f1']
    grid = GridSearchCV(clf, param_grid, iid = False, cv = 10, scoring = scoring, refit='f1')
    best_model = grid.fit(X_matrix, y_labels)

    alpha = best_model.best_estimator_.get_params()['alpha']
    precision = best_model.cv_results_['mean_test_precision'][best_model.best_index_]
    recall = best_model.cv_results_['mean_test_recall'][best_model.best_index_]
    fscore = best_model.cv_results_['mean_test_f1'][best_model.best_index_]

    print('Precision \t\t Recall  \t\t F-score  \t\t alpha')
    print(precision,'\t', recall,'\t', fscore,'\t', alpha, '\n')

    return precision, recall, fscore, alpha




def WriteFiles(file, tfidf, class_type):
    """
    Write files with the results of the LogReg, for different combinations of parameters
        - tf-idf matrices (tfidf = True) vs. term-freq count matrices (tfidf = False)
        - different values of min_df to build the matrices
        - different ngrams(used to create the features)
    """
    
    try:
        out_file = open(NB_PATH/('NaiveBayes_'+ file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(NB_PATH/('NaiveBayes_'+ file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score\n')
        
        for df in range(1+ngram, 21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F = CrossVal_NaiveBayes(X_matrix, y_labels)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) + '\t\t' + str(F) + '\n')
            
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()





""" ----------------------------- Single Run ----------------------------- """

def single_run_NB(file, ngram, df, tfidf, class_type):
    """
    Run CrossVal_NaiveBayes(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_NaiveBayes(X_matrix, y_labels)

# Uncomment next line to create only one classifier
#single_run_NB(DIET, ngram = 3, df = 23, tfidf = False, class_type = 'A')



def run_grid(file, ngram, df, tfidf, class_type):
    """
    Run grid_search(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    grid_search(X_matrix, y_labels)  

# Uncomment next line to run a grid search
#run_grid(DIET, ngram = 1, df = 1, tfidf = True, class_type = 'T')




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


# Uncomment next line to get the files in the Results/NaiveBayes folder
#multiple_runs_write_files(DIET)

