import requests
import xml.etree.ElementTree as ET

from Data_Collection.variables import *


def doi_to_pmid(dois):
    """
    Use DOIs to get PMIDs.

    Requires: dois a list with DOIs.
    Ensures: pmids, list with pmids retrieved from DOIs.
    """
    
    pmids_from_dois = []
    not_converted_dois = []

    # Use Pubmed converter to convert doi to pmid
    min_pos = 0
    csize = 200
    for i in range(int(len(dois)/csize)+1): # split dois in chunks of 200
        max_pos = len(dois)%csize + i*csize
        partial_dois = dois[min_pos:max_pos]
        min_pos = max_pos

        params = {"ids": ",".join(partial_dois),
                  "email": EMAIL,
                  "api_key": API_KEY}

        r = requests.get(DOI_PMID_URL, params)
        results = ET.fromstring(r.text)
        
        records = results.findall(".//record")
        for rec in records:
            pmid = rec.get("pmid")
            if pmid is not None:
                pmids_from_dois.append(pmid)
            if pmid is None:    # store dois without pmid
                not_converted_dois.append(rec.get("doi"))


    # Search for the remaining dois in Pubmed and try to find pmid 
    for doi in not_converted_dois:
        params = {"term": "{}[AID]".format(doi),
                  "db": "pubmed",
                  "email": EMAIL,
                  "api_key": API_KEY}

        status = True
        while status == True:
            try:
                r = requests.get(PUBMED_SEARCH_URL, params)
            except requests.exceptions.ConnectionError:
                return pmid
            if r.status_code != 200:
                print('{} API response: {}'.format(doi, r.status_code))
            else:
                results = ET.fromstring(r.text)
                try:
                    pmid = results.find(".//Id").text
                    pmids_from_dois.append(pmid)
                    
                except:
                    pass
                status = False
        
    return pmids_from_dois



def title_to_pmid(title_raw, author = ""):
    """
    Finding pmids using the title to search for the article in PubMed

    Requires: title_raw, str with the title of the article.
              author, str with the name of the authors. The first author will
              be used to improve the rearch.
    Ensures: list with pmids.
    """
    
    #error correction
    for key in ERRORS.keys():
        if key in title_raw:
            title_raw = title_raw.replace(key, ERRORS[key])

    
    title = title_raw + ' ' + author
    
    params =  {"term": "{}".format(title),
               "db": "pubmed",
               "email": EMAIL,
               "api_key": API_KEY}
    
    params_2 = {"term": "{}".format(title_raw),
                "field": "title",
                "db": "pubmed",
                "email": EMAIL,
                "api_key": API_KEY}

    pmid = None
    status = True

    while status == True:
        #try number 1: basic search with title + author
        try:
            r = requests.get(PUBMED_SEARCH_URL, params)
        except requests.exceptions.ConnectionError:
            return pmid
        if r.status_code != 200:
            #raise Exception('API response: {}'.format(r.status_code))
            print('{} API response: {}'.format(title, r.status_code))
        else:
            results = ET.fromstring(r.text)
            n_results = int(results.find(".//Count").text)

                   
            try: 
                if n_results == 1:
                    pmid = results.find(".//Id").text

                else:
                    try:
                        r_2 = requests.get(PUBMED_SEARCH_URL, params_2)
                    except requests.exceptions.ConnectionError:
                        return pmid
                    if r_2.status_code != 200:
                        #raise Exception('API response: {}'.format(r.status_code))
                        print('{} API response: {}'.format(title_raw, r_2.status_code))
                    else:
                        results_2 = ET.fromstring(r_2.text)
                        if int(results_2.find(".//Count").text) == 1:
                            pmid = results_2.find(".//Id").text
                    
                    #try number 2: remove stop words and search as title
                    if '[Title]' not in title_raw and pmid == None:
                        title2 = ' '.join([word for word in title_raw.split(' ') if word.lower() not in STOP_WORDS])
                        title2 = title2.replace(': ', ' ').replace(" ","[Title] AND ")
                        title2 = title2 + '[Title] ' + author + '[Author]'
                        pmid = title_to_pmid(title2)
                    
            except AttributeError:
                print("pmid not found for", title)
                   
            status = False

    return pmid



def get_pmids(pmids_from_dois, titles, authors):
    """
    Join the PMIDs retrieved from the DOIs and from the Titles in one single list.

    Requires: pmids_from_dois, list (from doi_to_pmid).
              titles and authors list (argument for the title_to_pmid function).
    Ensures: pmids, list with all single pmids.
    """

    #Get PMIDs from Titles
    pmids_from_titles = []
    n = 1
    author_pos = 0
    for t in titles:
        pmid = title_to_pmid(t, authors[author_pos])
        if pmid is not None:
            pmids_from_titles.append(pmid)
        n += 1
        author_pos += 1
    
    #Join PMIDs from Titles and from DOIs and remove duplicates
    total_pmids = pmids_from_titles + pmids_from_dois
    unique_pmids = list(set(total_pmids))

    print("Total number of pmids retrieved using DOIs: ", len(pmids_from_dois))
    print("Total number of pmids retrieved using Titles: ", len(pmids_from_titles))
    print("Total of unique pmids:", len(unique_pmids),'\n')
    
    return unique_pmids



def get_titles_abstracts(pmids):
    """
    Use the unique pmids to retrieve the articles' title and abstract from PubMed.

    Requires: pmids a list with all pmids.
    Ensures: titles, list with tuples.
             abstracts, list with tuples.
             Each tuple corresponds to one article. The first element is the pmid
             and the second one is the title or abstract.
    """
    
    titles = []
    abstracts = []

    min_pos = 0
    csize = 200
    for i in range(int(len(pmids)/csize)+1): # split dois in chunks of 200
        max_pos = len(pmids)%csize + i*csize
        partial_pmids = pmids[min_pos:max_pos]
        min_pos = max_pos   
        
        params = {"id": ",".join(partial_pmids),
                  "db": "pubmed",
                  "retmode": "xml",
                  "email": EMAIL,
                  "api_key": API_KEY}
            
        r = requests.get(PUBMED_FETCH_URL,params)
        results = ET.fromstring(r.text)

        for root in results.findall("PubmedArticle"):
            # PMIDs
            pmid = root.find(".//PMID").text
            
            # Title
            title = root.find(".//ArticleTitle")
            title = ''.join(title.itertext()) # remove tags
            title = title.replace("\n", " ").replace("\r", " ")
            titles.append((pmid, title))
            # Abstract
            if root.find(".//Abstract") is not None:
                abst_root = root.find(".//Abstract")
                all_abst = ""
                
                for abst in abst_root.iter('AbstractText'):
                    cur_text = ''.join(abst.itertext())
                    try:
                        all_abst += abst.attrib['Label'] + ': ' + cur_text + ' '
                    except:
                        all_abst = cur_text
                all_abst = all_abst.replace("\n", " ").replace("\r", " ")
                abstracts.append((pmid, all_abst))

    print("Total number of Titles:", len(titles))
    print("Total number of Abstracts:", len(abstracts))
    
    return titles, abstracts
    

    
def get_metadata(pmids):
    """
    Use the unique pmids to retrieve the articles' metadata from PubMed.

    Requires: pmids a list with all pmids.
    Ensures: metadata, list with tuples.
             Each tuple corresponds to one article. The first element is the pmid
             and the second one is the corresponding metadata.
    """

    metadata = []

    min_pos = 0
    csize = 200
    for i in range(int(len(pmids)/csize)+1): # split dois in chunks of 200
        max_pos = len(pmids)%csize + i*csize
        partial_pmids = pmids[min_pos:max_pos]
        min_pos = max_pos    

        params = {"id": ",".join(partial_pmids),
                  "db": "pubmed",
                  "retmode": "xml",
                  "email": EMAIL,
                  "api_key": API_KEY}
            
        r = requests.get(PUBMED_FETCH_META_URL,params)
        results = ET.fromstring(r.text)
        for root in results.findall("DocSum"):
            pmid = root.find('Id').text
            
            for item in root.findall('Item'):
                name = item.attrib['Name']
                if name == 'PubDate':
                    date = item.text[0:4]

                if name == 'AuthorList':
                    authors = ''
                    for author_name in item:
                        authors += author_name.text.replace(' ','').replace('-','') + ' '
                        
                if name == 'PmcRefCount':
                    cit_count = item.text
                if name == 'FullJournalName':
                    journal = item.text
                    
            metadata.append((pmid,date,authors[:-2],cit_count,journal))
        
    print("Total number of Metadata:", len(metadata), '\n')
    return metadata



