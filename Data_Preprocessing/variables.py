from pathlib import Path


""" ------------------------------ FILE NAMES ------------------------------ """

###########################################################################
################# Change the name(s) of your file(s) here #################
###########################################################################
                                                                     ######
# Individual file names                                              ######
DIET = ['diet']      # Do not write diet.bib                         ######
DBPS = ['dbps']                                                      ######
HCA = ['hca']                                                        ######
PAH = ['pah']                                                        ######
PCB = ['pcb']                                                        ######
PHTHALATES = ['phthalates']                                          ######
POLYBROMINATED = ['polybrominated']                                  ######
POLYCHLORINATED = ['polychlorinated']                                ######
REPRODUCIBILITY = ['reproducibility']                                ######
                                                                     ######
# All files names                                                    ######
ALL = ['dbps', 'diet', 'hca', 'pah', 'pcb', 'phthalates', \
       'polybrominated', 'polychlorinated', 'reproducibility']       ######                                        ######
                                                                     ######
# File with PMIDs from the relevant articles                         ######
REL_PMIDS_FILE = 'relevant_pubs_pmids.txt'                           ######
                                                                     ######
###########################################################################
###########################################################################



""" ----------------------------- FOLDER PATHS ----------------------------- """
# WOS Query Processed Results Folder Path
PROCESSED_DATA_PATH = Path('../Data_Collection/Processed_data')

# Data to use in the Machine Learning Model Folder Path
ML_PATH = Path('ML_data/')



""" --------------------------- FILE TERMINOLOGY --------------------------- """
# PMIDs
PMIDS_FILE = '_pmids.txt'

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


