import flair
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/content/drive/MyDrive/corpus1'

# column format indicating which columns hold the text and label(s)
column_name_map = {1: "text", 0: "label_topic"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=False,
                                         delimiter=',',    # tab-separated files
)
print(corpus)

label_dict = corpus.make_label_dictionary()

from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings
from flair.embeddings import FlairEmbeddings
embedding_types = [

    WordEmbeddings('hi'),
    FlairEmbeddings('path_to_embedding/best-lm.pt')
]

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
document_embeddings = DocumentRNNEmbeddings(embedding_types, hidden_size=256)

from flair.models import TextClassifier
from flair.trainers import ModelTrainer

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)
trainer.train('path_to_result',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=200,
              embeddings_storage_mode='gpu',
              train_with_dev=True)
