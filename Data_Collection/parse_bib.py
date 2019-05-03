import bibtexparser
from variables import *



def parse_bib(file):
    """
    Read .bib file

    Requires: .bib file
    Ensures: List with all the information from the .bib file
    """
    
    with open(RAW_DATA_PATH/(file + BIB_FILE), 'r', encoding = "utf8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    print("Total number of articles: ", len(bib_database.entries))
    return bib_database.entries



def get_dois(bib_data):
    """
    Get DOI from articles in the .bib file
    
    Requires: List with all the information from the .bib file (return
    from parse_bib)
    Ensures: List with all the DOIs from the articles in the .bib file
    """
    
    dois = []
    for article in bib_data:
        if "doi" in article:
            dois.append(article['doi'].replace("{", "").replace("}", ""))
    print("Total number of articles with DOI: ", len(dois))
    return dois



def get_titles(bib_data):
    """
    Get Titles from articles in the .bib file
    
    Requires: List with all the information from the .bib file (return from
    parse_bib)
    Ensures: List with all the Titles from the articles in the .bib file
    """
    
    titles = []
    for i in bib_data:
        if "title" in i:
            title = i["title"].strip().replace("\n", " ")
            titles.append(title.replace("{", "").replace("}", ""))
    print('Total number of articles with Titles:', len(titles),'\n')
    return titles



def get_authors(bib_data):
    """
    Get Authors from articles in the .bib file
    
    Requires: List with all the information from the .bib file (return from
    parse_bib)
    Ensures: List with all the Authors from the articles in the .bib file
    """
    
    authors = []
    for article in bib_data:
        if "author" in article:
            authors.append(article['author'][:article['author'].find(',')])

    return authors

    
