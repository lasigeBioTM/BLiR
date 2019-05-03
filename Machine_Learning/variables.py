from pathlib import Path
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')


""" ------------------------------ FILE NAMES ------------------------------ """

###########################################################################
################# Change the name(s) of your file(s) here #################
###########################################################################
                                                                     ######
# Individual file names                                              ######
FILE_1 = 'sample_file_1'     # Do not write sample_file_1.bib        ######
FILE_2 = 'sample_file_2'                                             ######
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


