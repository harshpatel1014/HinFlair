import flair
from flair.data import Corpus
from flair.datasets import UD_HINDI
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings

# 1. get the corpus
corpus: Corpus = UD_HINDI()
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),
    FlairEmbeddings('path_to_embedding/best-lm.pt')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('path_to_result',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
