import random
from itertools import chain
from collections import namedtuple, OrderedDict

Sentence = namedtuple("Sentence", "words tags")

class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
            
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class TrainingSet(namedtuple("_Trainingset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, training_corpus, cat ="all", tagset_dict = 'universal', train_split=0.8,seed=112890):
        
        if cat =="all" and tagset_dict =="all":
            tagged_sentences = training_corpus.tagged_sents()
        elif cat == "all":
            tagged_sentences = training_corpus.tagged_sents(tagset=tagset_dict)
        elif tagset_dict =="all":
            tagged_sentences = training_corpus.tagged_sents(categories=cat)
        else:
            tagged_sentences = training_corpus.tagged_sents(categories=cat,tagset=tagset_dict)
        word_sequences = list()
        tag_sequences = list()
 
        for sentence in tagged_sentences:
            word_sequence = list()
            tag_sequence = list()
            for word,tag in sentence:
                word_sequence.append(word)
                tag_sequence.append(tag)
            word_sequences.append(word_sequence)
            tag_sequences.append(tag_sequence)

        word_sequences = tuple(word_sequences)
        tag_sequences = tuple(tag_sequences)
        
        #prepares the wordset and tagset from the existing batch as its faster
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        
        keys = tuple(i for i in range(len(tagged_sentences)))
       
        #prepare the pair of word/tag sentences
        __sentences = list()
        _sentences = zip(word_sequences,tag_sequences)
        for _word,_tag in _sentences:
            __sentences.append(Sentence(words=_word,tags=_tag))
       
        sentences = OrderedDict(( (i, __sentences[i]) for i in range(len(keys)) ))
        
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))

        # split data into train/test sets after generating random keys 
        if seed is not None: random.seed(seed)
        random_keys = list(keys)
        random.shuffle(random_keys)
        split = int(train_split * len(random_keys))
        training_data = Subset(sentences, random_keys[:split])
        testing_data = Subset(sentences, random_keys[split:])

        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences)