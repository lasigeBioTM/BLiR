from pathlib import Path


""" ------------------------------ FILE NAMES ------------------------------ """

###########################################################################
################# Change the name(s) of your file(s) here #################
###########################################################################
                                                                     ######
# Individual file names                                              ######
DIET = 'diet'      # Do not write diet.bib                           ######
DBPS = 'dbps'                                                        ######
HCA = 'hca'                                                          ######
PAH = 'pah'                                                          ######
PCB = 'pcb'                                                          ######
PHTHALATES = 'phthalates'                                            ######
POLYBROMINATED = 'polybrominated'                                    ######
POLYCHLORINATED = 'polychlorinated'                                  ######
REPRODUCIBILITY = 'reproducibility'                                  ######
                                                                     ######
###########################################################################
###########################################################################

ALL = 'all'



""" ----------------------------- FOLDER PATHS ----------------------------- """
# Data to use in the Machine Learning Model Folder Path
ML_PATH = Path('../Data_Preprocessing/ML_data/')

# Algorithm Results
DT_PATH = Path('Results/DecisionTree/')
LG_PATH = Path('Results/LogReg/')
NB_PATH = Path('Results/NaiveBayes/')
NN_PATH = Path('Results/NeuralNetwork/')
RF_PATH = Path('Results/RandomForest/')
SVM_PATH = Path('Results/SVM/')



""" --------------------------- FILE TERMINOLOGY --------------------------- """
# Class
CLASS_FILE = '_class.txt'

# Titles
TITLES_FILE = '_titles.txt'

# Metadata
METADATA_FILE = '_metadata.txt'

# Abstract
ABSTRACTS_FILE = '_abstracts.txt'

# Titles + Metadata
TITLES_META_FILE = '_titles_meta.txt'



""" --------------------------- OTHER VARIABLES --------------------------- """

# downloaded from NLTK Corpora (http://www.nltk.org/nltk_data/)
STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', \
              "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', \
              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',\
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', \
              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', \
              'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', \
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',\
              'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', \
              'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', \
              'with', 'about', 'against', 'between', 'into', 'through', 'during', \
              'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', \
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',\
              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',\
              'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', \
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',\
              't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", \
              'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",\
              'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', \
              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',\
              'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',\
              "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
              'won', "won't", 'wouldn', "wouldn't"]

