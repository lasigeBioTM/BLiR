from pathlib import Path
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from transformers import Trainer, TrainingArguments

import torch
from seqeval.metrics import classification_report
DATA_DIR = "Data_Preprocessing/NER/"

def compute_metrics(pred):
    labels = pred.label_ids
    ner_labels = [] 
    for i in range(len(labels)):
      ner_labels.append([id2tag.get(x, "O") for x in labels[i]])
    preds = pred.predictions.argmax(-1)
    ner_preds = []
    for i in range(len(preds)):
      ner_preds.append([id2tag.get(x, "O") for x in preds[i]])
    #precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    #acc = accuracy_score(labels, preds)
    scores =classification_report(ner_labels, ner_preds,digits=4)
    scores = scores.split("\n\n")
    scores = scores[1].split()
    return {"p": float(scores[1]), "r": float(scores[2]), "f": float(scores[3])}


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    pmid_docs = []
    current_pmid = "0"
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            if line.startswith("#"):
              current_pmid = line.strip().split(" ")[-1]
              continue
            token, tag = line.split('\t')
            if len(token.strip()) == 0:
              continue
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
        pmid_docs.append(current_pmid)
        assert len(tokens) == len(tags)
    return token_docs, tag_docs, pmid_docs

def encode_tags(tags, encodings):
    truncated_seqs = 0
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for i, (doc_labels, doc_offset) in enumerate(zip(labels, encodings.offset_mapping)):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        
        # set labels whose first offset position is 0 and the second is not 0
        # in this case the tokens seq was truncated so we have unused labels
        if len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]) < len(doc_labels):
          #print(i, encodings[i].tokens, encodings[i].offsets)
          labels_i = 0
          for it in range(len(doc_offset)):
            if encodings[i].offsets[it][0] == 0 and encodings[i].offsets[it][1] != 0:
              #print(encodings[i].tokens[it], encodings[i].offsets[it], doc_labels[labels_i])
              doc_enc_labels[it] = doc_labels[labels_i]
              labels_i += 1
          #print(doc_labels[labels_i:])
          truncated_seqs += 1
          #import pdb; pdb.set_trace()
        else:
          # assign labels to tokens that start with zero (first subword) and do not end in zero (special tokens)
          doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    print("truncated sequences:", truncated_seqs)
    return encoded_labels

class BlirDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, pmids):
        self.encodings = encodings
        self.labels = labels
        self.pmids = pmids # pmids of sentences

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        #item['pmid'] = self.pmids[idx]
        return item

    def __len__(self):
        return len(self.labels)

texts, tags, pmid_docs = read_wnut(DATA_DIR + 'biomarker_entities_all.conll')

train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=0.2)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
print(unique_tags)

tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=128)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=128)

# deal with subword tokenization problem

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


if "offset_mapping" in train_encodings:
  train_encodings.pop("offset_mapping") # we don't want to pass this to the model
if "offset_mapping" in val_encodings:
  val_encodings.pop("offset_mapping")
train_dataset = BlirDataset(train_encodings, train_labels, ["0"]*len(train_labels))
val_dataset = BlirDataset(val_encodings, val_labels, ["0"]*len(train_labels))
model = BertForTokenClassification.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', num_labels=len(unique_tags))


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=350,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    save_total_limit=3
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

#RUrun best epoch on positive docs
test_texts, test_tags, pmid_docs = read_wnut(DATA_DIR + 'documents_to_annotate.conll')
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=128)
test_labels = encode_tags(test_tags, test_encodings)
if "offset_mapping" in test_encodings:
  test_encodings.pop("offset_mapping") # we don't want to pass this to the model
test_dataset = BlirDataset(test_encodings, test_labels, pmids=pmid_docs)
pred = trainer.predict(test_dataset)
print(pmid_docs)

#  generate full entity names
preds = pred.predictions.argmax(-1)
ner_preds = []
for i in range(len(preds)):
  sentence_tags = [id2tag.get(x, "O") for x in preds[i]]
  new_entity = []
  previous_tag = ""
  #import pdb; pdb.set_trace()
  for it, tag in enumerate(sentence_tags):
    if tag == "B" or tag == "I":
      #print(test_encodings[i].tokens[it])
      #new_entity += " " + test_encodings[i].tokens[it]
      new_entity.append(test_encodings[i].ids[it])
    elif tag == "O" and previous_tag == "I" or previous_tag == "B":
      new_entity = tokenizer.decode(new_entity)
      new_entity = new_entity.replace(" - ", "-").replace(" / ", "/").replace(" ( ", "(").replace(" ) ", ")").replace(" [ ", "[").replace(" ]", "]")
      if not new_entity.startswith("#") and not new_entity.startswith("[") and len(new_entity.strip()) > 3:
        ner_preds.append((new_entity, test_dataset.pmids[i]))
        #print(new_entity)
      new_entity = []
    previous_tag = tag

entity_to_doc = {}
for e in ner_preds:
  if e[0] not in entity_to_doc:
    entity_to_doc[e[0]] = set()
  entity_to_doc[e[0]].add(e[1])
sorted_entities = {k: v for k, v in sorted(entity_to_doc.items(), key=lambda item: len(item[1]), reverse=True)}

#open reference biomarkers
known_biomarkers = set()
with open("publications_compounds.txt") as f:
  for line in f:
    biomarker = line.strip().split("\t")[3].lower()
    known_biomarkers.add(biomarker)

#write results to file
e_count = 0
results_file = open(DATA_DIR + "entity_results10epoch5-e5bs32.txt", 'w')
for e in sorted_entities:
  if e not in known_biomarkers and e.replace(" ", "") not in known_biomarkers:
    print(e, sorted_entities[e])
    results_file.write(f"ENTITY:{e}\n")
    for pmid in list(sorted_entities[e]):
        results_file.write(f"PUBMED_ID:{pmid}\n")
    results_file.write("\n")
    e_count += 1
  if e_count > 100:
    break
results_file.close()