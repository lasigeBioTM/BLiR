# BLiR
Biomarker Literature Retrieval

This tool allows to:
1. Retrieve abstracts, titles and metadata (publication date, author names, number of times the article was cited and journal name) from PubMed by either using the article's PMID or BibTeX entry;
2. Preprocess text data into numerical data;
3. Build machine learning models (classifiers), with a supervised learning approach. The classifiers can predict an article's relevance based on its abstract, title or metadata.


## Dependencies
- Python 3.6
- requirements.txt. Run `pip install -r requirements.txt`
- Stopword list from [NLTK package](https://www.nltk.org/). Run `python -m nltk.downloader stopwords`
- NCBI account and API key. Please check [NCBI API Keys](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/) for more information.

## Preparing the Data
In order for the program to work, you need to provide:
1. BibTeX file(s) or text file(s) with PMIDs;
2. A text file with the PMIDs of the relevant articles.

All the folders you need are already created, so you do not have to change any folder paths. Just place the files in the assigned folders and change the names of the variables, as instructed below.

#### BibTeX file:
If you have a `.bib` file, place it in the [Raw_data](Data_Collection/Raw_data) folder.

#### Text file with PMIDs:
If you have a `.txt` file with PMIDs, place it in the [Processed_data](Data_Collection/Processed_data) folder.

#### Text file with the relevant PMIDs:
Place your `.txt` file with the PMIDs of the relevant articles in the [Data_Preprocessing](Data_Preprocessing) folder.

See the sample files and match your files to that format. After placing them in the apropriate location, go to the `variables.py` script (there are three of them, one in each folder) and change the variables to match the name of your files.

## Data Collection
If you have a `.bib` file, run the first option in the `get_pmids_titles_abst_meta.py` script in order to get all the files in the [Processed_data](Data_Collection/Processed_data) folder. For example, if you have a file called 'cancer.bib' and you run the first option, you will get the following files: 'cancer_abstracts.txt', 'cancer_metadata.txt', 'cancer_pmids.txt' and 'cancer_titles.txt'.

If you have a `.txt` file with PMIDs, run the second option in the `get_pmids_titles_abst_meta.py` script in order to get the remaining three files in the [Processed_data](Data_Collection/Processed_data) folder. For example, if you already have a file called 'cancer_pmids.txt' and you run this option, you will get the following files: 'cancer_abstracts.txt', 'cancer_metadata.txt' and 'cancer_titles.txt'.

## Data Preprocessing
Run `get_ML_data.py` to get all the data you need to train the models in the [ML_data](Data_Preprocessing/ML_data) folder.

## Train the models
In the [Machine_Learning](Machine_Learning) folder, choose the script with the algorithm you want to run: `DecisionTree.py`, `LogRegression.py`, `NaiveBayes.py`, `NeuralNetwork.py`, `RandomForest.py` or `SupportVectorMachine.py`. Inside each script there are further instructions and options, select the ones are more suitable for you.
