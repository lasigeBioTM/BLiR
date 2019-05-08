from pathlib import Path



""" ------------------------------ FILE NAMES ------------------------------ """

###########################################################################
################# Change the name(s) of your file(s) here #################
###########################################################################
                                                                     ######
# Individual file names                                              ######
FILE_1 = ['sample_file_1']     # Do not write sample_file_1.bib      ######
FILE_2 = ['sample_file_2']                                           ######
                                                                     ######
# All files names                                                    ######
ALL = ['sample_file_1', 'sample_file_2']                             ######
                                                                     ######
###########################################################################
###########################################################################



""" -------------------------- PUBMED CREDENTIALS -------------------------- """

###########################################################################
####################### Change your credentials here ######################
###########################################################################
                                                                     ######
EMAIL = ""                                                           ######
API_KEY = ""                                                         ######
                                                                     ######
###########################################################################
###########################################################################



""" ----------------------------- FOLDER PATHS ----------------------------- """
# WOS Query Results Folder Path
RAW_DATA_PATH = Path('Raw_data/')

# WOS Query Processed Results Folder Path
PROCESSED_DATA_PATH = Path('Processed_data/')



""" --------------------------- FILE TERMINOLOGY --------------------------- """

# bib
BIB_FILE = '.bib'

# PMIDs
PMIDS_FILE = '_pmids.txt'

# Titles
TITLES_FILE = '_titles.txt'

# Metadata
METADATA_FILE = '_metadata.txt'

# Abstract
ABSTRACTS_FILE = '_abstracts.txt'

# Class
CLASS_FILE = '_class.txt'



""" --------------------------------- URLS --------------------------------- """

DOI_PMID_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_FETCH_META_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'



""" --------------------------- OTHER VARIABLES ---------------------------- """

# Some articles were not being found on PubMed due to some words in the title 
#not being writen in the same way as they were in the .bib files from WOS.

# This specific error correction is for the Exposome-Explorer dataset. You can 
#change it to match your data, if you find any errors.

ERRORS = {'epsilon dA':'epsilondA','metaholites':'metabolites', \
          'polyhrominated':'polybrominated', 'Causcasian':'Caucasian', \
          'SEMIQUANTITATIVE':'SEMI-QUANTITATIVE', '.1.':'I', ' .2.':'. II.',\
          ' .3.':'. III.', 'K-1':'K1', 'between 2 dietary':'between two dietary',\
          'NONSMOKERS':'NON-SMOKERS', 'Immunological measurement of polycyclic aromatic'\
          : 'Immunologic measurement of polycyclic aromatic'}


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


