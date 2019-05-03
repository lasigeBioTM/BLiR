from variables import *


def write_file_pmids(file, pmids):
    """
    Created a file with the PMIDs from the DOIs

    Requires: pmids list with PMIDs (return from get_pmids(pmids_from_dois, titles, authors))
              file str with name from .bib file
    Ensure: .txt file with DOIs
    """
    
    file_pmids = open(PROCESSED_DATA_PATH / (file + PMIDS_FILE), 'w', encoding = 'utf-8')
    
    for pmid in pmids:
        file_pmids.write(pmid + '\n')

    file_pmids.close()



def write_file_title(file, titles):
    """
    Created a file with the pmids and corresponding title

    Requires: titles (return from get_titles_abstracts(pmids))
              file str with name from .bib file
    Ensure: .txt file with title
    """
    
    file_titles = open(PROCESSED_DATA_PATH / (file + TITLES_FILE), 'w', encoding = 'utf-8')

    for title in titles:
        file_titles.write(title[0] + '\t' + title[1] + '\n')

    file_titles.close()
    


def write_file_abstract(file, abstracts):
    """
    Created a file with the pmids and corresponding abstract

    Requires: abstracts (return from get_titles_abstracts(pmids))
              file str with name from .bib file
    Ensure: .txt file with abstracts
    """

    file_abstracts = open(PROCESSED_DATA_PATH / (file + ABSTRACTS_FILE), 'w', encoding = 'utf-8')
    for abstract in abstracts:
        file_abstracts.write(abstract[0] + '\t' + abstract[1] + '\n')

    file_abstracts.close()



def write_file_metadata(file, metadata):
    """
    Created a file with the pmids and corresponding metadata

    Requires: metadata (return from get_metadata(pmids))
              file str with name from .bib file
    Ensure: .txt file with metadata
    """
    
    file_metadata = open(PROCESSED_DATA_PATH / (file + METADATA_FILE), 'w', encoding = 'utf-8')
    for meta in metadata:
        file_metadata.write(meta[0]+'\t'+meta[1]+'\t'+meta[2]+'\t'+meta[3]+'\t'+meta[4]+'\n')

    file_metadata.close()



