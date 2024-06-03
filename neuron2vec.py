import numpy as np
from gensim.models import Word2Vec
from connectome_tools.connectome_loaders import load_flywire
from get_sentences import SentenceGenerator
from tqdm import tqdm

data_path = "../data/"
neurons, J, nts_Js = load_flywire(data_path, by_nts=True, include_spatial=False)
sentence_gen = SentenceGenerator(J,
                                    pre_post_neurons=10,
                                    batch_size=32,
                                    max_sequence_length=10)

sentences = sentence_gen.random_walk_gen()

def training(epochs=10):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    for i in tqdm(range(epochs)):
        new_sentences = sentence_gen.random_walk_gen()
        model.build_vocab(new_sentences, update=True)
        model.train(new_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        
    model.save("updated_neuron2vec.model")

    return model

model = training()
#model = Word2Vec.load("updated_word2vec.model")
