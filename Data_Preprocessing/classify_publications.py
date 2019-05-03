from variables import *



def get_list_all_pmids(files):
    """
    Read _pmids.txt file(s) to get all the pmids from the publications

    Requires: files: a list with the name(s) of the file(s).
    Ensures: all_pmids a list with all pmids.
    """
    
    all_pmids = []
    for file in files:
        in_file = open(PROCESSED_DATA_PATH / (file + PMIDS_FILE), 'r', encoding = "utf8")
        pmids = [pmid[:-1] for pmid in in_file.readlines()]
        in_file.close()
        all_pmids.extend(pmids)

    all_pmids = set(all_pmids)

    return all_pmids



def get_relevant_pmids():
    """
    Use the file with all relevant publications to get a list with relevant pmids
    used to built the database

    Ensures: relevant_pmids a list with all pmids from relevant publications.
    """
    
    in_file = open(REL_PMIDS_FILE, 'r', encoding = "utf8")
    relevant_pmids = [pmid[:-1] for pmid in in_file.readlines()]
    in_file.close()
    
    return relevant_pmids



def get_irrelevant_pmids(all_pmids,relevant_pmids):
    """
    Get the pmids from the irrelevant publications

    Requires: all_pmids a list with pmids from all publications;
              relevant_pmids a list with all pmids from relevant publications.
    Ensures: irrelevant_pmids a list with all pmids from irrelevant publications.
    """
        
    irrelevant_pmids = list(set(all_pmids)-set(relevant_pmids))

    return irrelevant_pmids



def get_all_pmids_labels(files):
    """
    Map pmids to the respective label

    Requires: files a list with the name(s) of the file(s).
    Ensures: all_pmids_labels a list with tuples (pmid, label).
    """
    
    print("................................ Query Results ................................")
    
    all_pmids = get_list_all_pmids(files)
    relevant_pmids = get_relevant_pmids()
    irrelevant_pmids = get_irrelevant_pmids(all_pmids, relevant_pmids)
    
    print('Number of pubs:', len(all_pmids))
    print('Number of relevant pubs:', len(all_pmids)-len(irrelevant_pmids))
    print('Number of irrelevant pubs:', len(irrelevant_pmids), '\n') 


    all_pmids_labels = []    
    for pmid in all_pmids:
        if pmid in relevant_pmids:
            all_pmids_labels.append((pmid, '1'))
        else:
            all_pmids_labels.append((pmid, '0'))
            
    return all_pmids_labels


