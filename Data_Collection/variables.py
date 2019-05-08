from pathlib import Path
from nltk.corpus import stopwords



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

ERRORS = {'epsilon dA':'epsilondA','metaholites':'metabolites', \
          'polyhrominated':'polybrominated', 'Causcasian':'Caucasian', \
          'SEMIQUANTITATIVE':'SEMI-QUANTITATIVE', '.1.':'I', ' .2.':'. II.',\
          ' .3.':'. III.', 'K-1':'K1', 'between 2 dietary':'between two dietary',\
          'NONSMOKERS':'NON-SMOKERS', 'Immunological measurement of polycyclic aromatic'\
          : 'Immunologic measurement of polycyclic aromatic'}

STOP_WORDS = stopwords.words('english')


