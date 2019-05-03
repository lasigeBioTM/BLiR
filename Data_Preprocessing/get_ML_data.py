from variables import *
from classify_publications import get_all_pmids_labels


def get_class(files):
    """
    Gets ordered pmids and corresponding classes

    Requires: files: list with the name(s) of the file(s).
    Ensures: all_classes: an ordered list with tuples (pmid, label).
    """
    
    all_classes = get_all_pmids_labels(files)

    all_classes.sort(key = lambda tup: int(tup[0]))
    
    return all_classes



def get_individual_info(info, files):
    """
    Gets titles, abstracts or metadata one or several files
    
    Requires: info: TITLES_FILE or ABSTRACTS_FILE or METADATA_FILE;
              files: list with the name(s) of the file(s).
    Ensures: info_pmids_dict: dict with pmids as keys and the corresponding
    title, abstract or metadata as values.
    """

    info_pmids_dict = {}

    for file in files:
        in_file = open(PROCESSED_DATA_PATH/(file + info),'r',encoding = 'utf-8')
        for pub in in_file.readlines():
            pub = pub[:-1].split('\t',1)
            if pub[0] not in info_pmids_dict.keys():
                info_pmids_dict[pub[0]] = pub[1]
        in_file.close()
        
    return info_pmids_dict



def get_all_info(files):
    """
    Gather all information from all files.
    
    Requires: files: list with the name(s) of the file(s).
    Ensures: list with tuples. Each element of the tuple corresponds
    to the pmid, class, title, abstract and metadata of one article.
    """

    all_info = []

    all_classes = get_class(files)
    all_titles_pmids = get_individual_info(TITLES_FILE, files)
    all_abstracts_pmids = get_individual_info(ABSTRACTS_FILE, files)
    all_metadata_pmids = get_individual_info(METADATA_FILE, files)

    for article in all_classes:
        pmid = article[0]

        #only save pmids with titles, abstracts and metadata
        if pmid in all_abstracts_pmids.keys():  
            classe = article[1]
            title = all_titles_pmids[pmid]
            abstract = all_abstracts_pmids[pmid]
            metadata = all_metadata_pmids[pmid]

            all_info.append((pmid, classe, title, abstract, metadata))

    return all_info



def write_files(all_info, file):
    """
    Writes all .txt files needed to apply the machine learning Model
    
    Requires: all_info: list with tuples (return from get_all_info(files)).
    Ensures: 4 files (classes, abstracts, titles and metadata).
    """

    class_file = open(ML_PATH / (file + CLASS_FILE), 'w', encoding = 'utf-8')
    titles_file = open(ML_PATH / (file + TITLES_FILE), 'w', encoding = 'utf-8')
    abst_file = open(ML_PATH / (file + ABSTRACTS_FILE), 'w', encoding = 'utf-8')
    meta_file = open(ML_PATH / (file + METADATA_FILE), 'w', encoding = 'utf-8')    

    for pub in all_info:
        class_file.write(pub[1] + '\n')
        titles_file.write(pub[2] + '\n')
        abst_file.write(pub[3] + '\n')
        meta_file.write(pub[4] + '\n')

    class_file.close()
    titles_file.close()
    abst_file.close()
    meta_file.close()

    

def __main__(files):
    """ Main Function """

    if len(files) == 1:
        all_info = get_all_info(files)
        write_files(all_info, files[0])

    if len(files) > 1:
        all_info = get_all_info(files)
        write_files(all_info, file = 'all')
    


# Join all absts, titles and metadata from file_1 and file_2 in the same files.
#__main__(ALL)

#OR

# Create individual absts, titles and metadata files for file_1 and file_2.
#__main__(FILE_1)
#__main__(FILE_2)


