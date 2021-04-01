from pathlib import Path
import spacy
from spacy.tokens import Span
from spacy_conll import init_parser
from spacy_conll import ConllFormatter
import merpy
from Data_Collection import query_pubmed
import sys
sys.path.append('../')

""" Annotate all documents with all entities """

nlp = spacy.load("en_core_web_sm")
#conllformatter = ConllFormatter(nlp)
#nlp.add_pipe(conllformatter, last=True)

output_file = open("biomarker_entities_all.conll", 'w')
# generate ConLL style file using the publication_compounds.txt
pmid_to_abst = {}


#first read all title
with Path("Data_Collection/Processed_data/all_pmid_titles.txt").open(encoding="utf8") as abs_file:
    for line in abs_file:
        values = line.split("\t")
        if len(values) > 1:
            pmid_to_abst[values[0]] = values[1].strip().lower()

#first read all abstracts
with Path("Data_Collection/Processed_data/all_pmid_abstracts.txt").open(encoding="utf8") as abs_file:
    for line in abs_file:
        values = line.split("\t")
        if len(values) > 1:
            if values[0] not in pmid_to_abst:
                pmid_to_abst[values[0]] = values[1].strip().lower()
            else:
                pmid_to_abst[values[0]] += ". " +  values[1].strip().lower()

missing_pmids = []
all_entities = set()
with open("publications_compounds.txt") as compounds_file:
    next(compounds_file)
    for line in compounds_file:
        values = line.strip().split("\t")
        pmid = values[1]
        all_entities.add(values[3].lower())
        if pmid not in pmid_to_abst:
            missing_pmids.append(pmid)

merpy.create_lexicon(all_entities, "biomarkers")
merpy.process_lexicon("biomarkers")

#recover missing pmids:
titles, abstracts = query_pubmed.get_titles_abstracts(missing_pmids)
for t in titles:
    pmid_to_abst[t[0]] = t[1].lower()
for a in abstracts:
    if a[0] not in pmid_to_abst:
        pmid_to_abst[a[0]] = a[1].lower()
    else:
        pmid_to_abst[a[0]] += ". " + a[1].lower()


missing_texts = 0
total_entities = 0
total_sents = 0
total_docs = 0
# can be parallelized
for pmid in pmid_to_abst:
    if pmid == "":
        continue
    if pmid not in pmid_to_abst:
        print("missing this abstract:", pmid)
        missing_texts += 1
        continue
    total_docs += 1
    doc = nlp(pmid_to_abst[pmid])
    doc_entities = merpy.get_entities(pmid_to_abst[pmid], "biomarkers")
    entity_spans = []
    for e in doc_entities:
        try:
            int(e[0]), int(e[1])
        except ValueError:
            print("ERROR", e)
            continue
        entity_spans.append(doc.char_span(int(e[0]), int(e[1]), label="GPE"))
    entity_spans = [e for e in entity_spans if e is not None]
    try:
        doc.ents = entity_spans[:]
    except:
        import pdb; pdb.set_trace()
        continue
    total_entities += len(entity_spans)
    for sent in doc.sents:
        if len(sent.ents) > 0:
            total_sents += 1
            for t in sent:
                output_file.write(f"{t.text}\t{t.ent_iob_}\n")
            output_file.write("\n")

output_file.close()
print("total entities", total_entities, "total sents", total_sents, "total docs", total_docs)
print("missing", missing_texts, "pmids")


# total entities 23729 total sents 15296 total docs 7331

