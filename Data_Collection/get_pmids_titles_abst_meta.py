from variables import *
from parse_bib import *
from query_pubmed import *
from write_files import *
from pathlib import Path


""" ----------------------- STEP 1: PARSE .BIB FILE ----------------------- """

def parsebib(file):
    
    # Get data from .bib file
    bib_data = parse_bib(file)

    # Get DOIs
    dois = get_dois(bib_data)

    # Get Authors
    authors = get_authors(bib_data)

    # Get Titles
    titles = get_titles(bib_data)
            
    return dois, titles, authors




""" ------------------------- STEP 1: QUERY PUBMED ------------------------ """

def query_pubmed_pmids(dois, titles, authors):
            
    # Get PMIDs from DOIs
    pmids_from_dois = doi_to_pmid(dois)
            
    # Get PMIDs from Titles and unique PMIDS from both Titles and DOIs
    pmids = get_pmids(pmids_from_dois, titles, authors)

    return pmids


def query_pubmed_abst_meta(pmids):
            
    #Get PubMed Metadata Titles and Abstracts
    titles, abstracts = get_titles_abstracts(pmids)

    # Get PubMed Metadata
    metadata = get_metadata(pmids)
    
    return titles, abstracts, metadata




""" --------------------------- MAIN FUNCTIONS ---------------------------- """

def main_bib(files):
    
    for file in files:
        print('\n' + file.upper())
                
        print('...................... Parsing .bib File .....................')
        dois, titles, authors = parsebib(file)

        print('........................ Getting pmids .......................')
        pmids = query_pubmed_pmids(dois, titles, authors)

        print('.................... Writing PMIDs Files .....................\n')
        write_file_pmids(file, pmids)

        print('.......... Getting titles, abstracts and metadata ............')
        titles, abstracts, metadata = query_pubmed_abst_meta(pmids)

        print('........ Writing Titles, Abstracts and Metadata Files ........\n')
        write_file_title(file, titles)
        write_file_abstract(file, abstracts)
        write_file_metadata(file, metadata)
        


def main_pmids(files):
    
    for file in files:
        print('\n' + file.upper())
                
        print('.......... Getting titles, abstracts and metadata ............')
        pmids = [pmid[:-1] for pmid in open(PROCESSED_DATA_PATH / (file\
                                                  + PMIDS_FILE), 'r', encoding = 'utf-8')]

        titles, abstracts, metadata = query_pubmed_abst_meta(pmids)

        print('........ Writing Titles, Abstracts and Metadata Files ........\n')
        write_file_title(file, titles)
        write_file_abstract(file, abstracts)
        write_file_metadata(file, metadata)


###################################### OPTION 1 ######################################
# Uncomment one of the next lines if you have a .bib file in the Raw_data folder
#main_bib(ALL)          # run for file1 and file2
#OR
#main_bib(FILE_1)       # single run for file1
#main_bib(FILE_2)       # single run for file2


###################################### OPTION 2 ######################################
# Uncomment one of the next lines if you have a _pmids.txt file in the
#Processed_data folder
#main_pmids(ALL)          # run for file1 and file2
#OR
#main_pmids(FILE_1)       # single run for file1
#main_pmids(FILE_2)       # single run for file2


