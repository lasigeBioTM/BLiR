from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import *



def CrossVal_NeuralNetwork(X_matrix, y_labels):
    """
    Neural Network using Cross Validation

    Requires: X_matrix: a doc-term count or tfidf matrix;
              y_labels: a matrix with labels.
    Ensures: Precision, Recall and F-score for class 1 (relevant).
    """

    clf = MLPClassifier(solver = 'lbfgs', random_state = 0)
    
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
        out_file = open(NN_PATH/('NeuralNetwork_'+ file + '_' + class_type+'.txt'),'a')
    except:
        out_file = open(NN_PATH/('NeuralNetwork_'+ file + '_' + class_type+'.txt'),'w')
    
    out_file.write('\n------------------------------- TF-IDF '+ str(tfidf).upper() +' -------------------------------\n')

    for ngram in range(1,4):
        out_file.write('n-grams(1-'+str(ngram)+')\n' + 'df\t\t Precision \t\t\t\t Recall \t\t\t\t F-score \n')
        
        for df in range(1+ngram,21+ngram):
            X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
            P,R,F = CrossVal_NeuralNetwork(X_matrix, y_labels)
            out_file.write(str(df) + '\t' + str(P) + '\t\t' + str(R) + '\t\t' + str(F) + '\t\t' + '\n')
            
        out_file.write('\n')
        
    out_file.write('\n\n')
    out_file.close()




""" ----------------------------- Single Run ----------------------------- """

def single_run_CV(file, ngram, df, tfidf, class_type):
    """
    Run CrossVal_NeuralNetwork(X_matrix, y_labels)
    """
    X_matrix, y_labels = build_matrices(file, ngram, df, tfidf, class_type)
    CrossVal_NeuralNetwork(X_matrix, y_labels)

# Uncomment next line to create only one classifier
#single_run_CV(file, ngram = 3, df = 16, tfidf = False, class_type = 'A')




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


# Uncomment next line to get the files in the Results/NeuralNetwork folder
#multiple_runs_write_files(file)

