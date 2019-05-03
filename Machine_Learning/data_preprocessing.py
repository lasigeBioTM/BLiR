import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from variables import *



def labels_matrix(class_file):
    """
    CREATE LABELS MATRIX

    Requires: class_file: str with name of a .txt file with the labels of each article.
    Ensures: y_labels: matrix with the labels of each article.
    """
    
    list_labels = [int(label[:-1]) for label in open(class_file)]
    y_labels = np.array(list_labels)

    return y_labels



def build_title_or_abst_matrix(data_file, ngram, df, tfidf):
    """
    CREATE TITLES OR ABSTRACT MATRIX

    Requires: data_file: str with name of a .txt file with the titles or abstracts of each article;
              ngram (ngram_range(1,ngram)): int; 
              df (min_dif): int or float;
              tfidf: True(TFIDF matrix) or False(term-count matrix).
    Ensure: X_matrix: term-count or TFIDF matrix.
    """

    # --------------- CREATE [n_samples, n_features] SCYPI.SPARSE MATRIX ----------------

    corpus = open(data_file, 'r', encoding = 'utf-8').readlines()

    for i in range(len(corpus)):
        corpus[i] = corpus[i].split(' ')
        for j in range(len(corpus[i])):
            corpus[i][j] = SnowballStemmer('english').stem(corpus[i][j])
        corpus[i] = ' '.join(corpus[i])


    # Stop Word Removal and Tokenization
    if tfidf == False:
        vectorizer = CountVectorizer(min_df = df, stop_words = STOP_WORDS, ngram_range=(1,ngram))
    else:
        vectorizer = TfidfVectorizer(min_df = df, stop_words = STOP_WORDS, ngram_range=(1,ngram))


    # Built Document-term matrix
    X = vectorizer.fit_transform(corpus)
    X_matrix = X.toarray()  

    print('Features:', X_matrix.shape, '\n')
    
    return X_matrix



def build_meta_matrix(data_file, ngram, df, tfidf):
    """
    BUILD METADATA MATRIX

    Requires: data_file: str with name of a .txt file with the metadata of each article;
              ngram (ngram_range(1,ngram)): int; 
              df (min_dif): int or float;
              tfidf: True(TFIDF matrix) or False(term-count matrix).
    Ensure: X_matrix: term-count or TFIDF matrix.
    """

    corpus = open(data_file, 'r', encoding = 'utf-8').readlines()
    for i in range(len(corpus)):
        corpus[i] = corpus[i].split('\t')

    # YEAR
    year = [int(line[0]) for line in corpus]
    year_matrix = np.array(year)

    # AUTHOR
    author = [(line[1]) for line in corpus]
    if tfidf == False:
        author_matrix = CountVectorizer(min_df = df).fit_transform(author).toarray()
    else:
        author_matrix = TfidfVectorizer(min_df = df).fit_transform(author).toarray()

    # CITATION COUNT
    CitCount = [int(line[2]) for line in corpus]
    CitCount_matrix = np.array(CitCount)

    # JOURNAL
    journal = [(line[3]) for line in corpus]
    if tfidf == False:
        journal_vect = CountVectorizer(min_df = df, stop_words = STOP_WORDS, ngram_range=(1,ngram))
    else:
        journal_vect = TfidfVectorizer(min_df = df, stop_words = STOP_WORDS, ngram_range=(1,ngram))
    journal_matrix = journal_vect.fit_transform(journal).toarray()

    # JOIN MATRICES
    meta_matrix = np.column_stack((year_matrix, author_matrix, CitCount_matrix, \
                                journal_matrix))
    
    print('Features:', meta_matrix.shape, '\n')
    
    return meta_matrix




def build_matrices(file, ngram, df, tfidf, class_type):
    """
    BUILD X_MATRIX AND Y_MATRIX

    Requires: data_file: str with name of a .txt file with the titles, abstracts or metadata of each article;
              ngram (ngram_range(1,ngram)): int;
              df (min_dif): int or float;
              tfidf: True(TFIDF matrix) or False(term-count matrix);
              class_type = 'T'(Titles), 'A'(Abstracts), 'M'(Metadata) or 'TM' (Titles + Metadata).
    Ensure: X_matrix, y_labels.
    """

    print('\n', class_type, '   ngram:', ngram,'   df:', df, '   Tf-idf:', tfidf)
    y_labels = labels_matrix(ML_PATH/(file + CLASS_FILE))

    if class_type == 'T':
        X_matrix = build_title_or_abst_matrix(ML_PATH/(file + TITLES_FILE), ngram, df, tfidf)

    elif class_type == 'A':
        X_matrix = build_title_or_abst_matrix(ML_PATH/(file + ABSTRACTS_FILE), ngram, df, tfidf)

    elif class_type == 'M':
        X_matrix = build_meta_matrix(ML_PATH/(file + METADATA_FILE), ngram, df, tfidf)
        
    elif class_type == 'TM':
        title_matrix = build_title_or_abst_matrix(ML_PATH/(file + TITLES_FILE), ngram, df, tfidf)
        meta_matrix = build_meta_matrix(ML_PATH/(file + METADATA_FILE), ngram, df, tfidf)
        X_matrix = np.column_stack((title_matrix, meta_matrix))

    return X_matrix, y_labels



