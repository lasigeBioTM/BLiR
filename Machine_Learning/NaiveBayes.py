from sklearn.naive_bayes import MultinomialNB
from variables import *
from GenericClassifier import BlirModel


param_grid = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}
gnb = MultinomialNB(alpha=0.01)


model = BlirModel(gnb, "NaiveBayes")

# Uncomment next line to create only one classifier
model.single_run(DIET, ngram=3, df=23, tfidf=False, class_type="A")


# Uncomment next line to run a grid search
# run_grid(DIET, ngram = 1, df = 1, tfidf = True, class_type = 'T')


# Uncomment next line to get the files in the Results/NaiveBayes folder
# multiple_runs_write_files(DIET)
