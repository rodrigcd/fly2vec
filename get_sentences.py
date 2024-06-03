import numpy as np
from scipy.special import softmax
import time


class SentenceGenerator(object):

    def __init__(self, J, pre_post_neurons=10, batch_size=32, max_sequence_length=50):
        self.J = J
        self.pre_post_neurons = pre_post_neurons
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

    def batch_sentence(self, J, batch_size=32, pre_post_neurons=10):
        n_neurons = self.J.shape[0]
        sampled_neurons = np.random.choice(n_neurons, batch_size, replace=False)
        sentences = []
        target = []
        for i, neuron_index in enumerate(sampled_neurons):
            pre_synaptic = J[neuron_index, :].toarray()
            pre_synaptic = pre_synaptic[0, :]
            post_synaptic = J[:, neuron_index].toarray()
            post_synaptic = post_synaptic[:, 0]
            stronger_pre = np.argsort(pre_synaptic)[-pre_post_neurons:]
            stronger_post = np.argsort(post_synaptic)[-pre_post_neurons:]
            sentence = np.concatenate((stronger_pre, stronger_post)).astype(int)
            sentences.append(sentence.tolist())
            target.append(neuron_index)
        return sentences, target

    def random_walk_gen(self):
        starting_nodes = np.random.choice(self.J.shape[0], self.batch_size, replace=False)
        batch_data = []
        batch_data.append(starting_nodes)
        for i in range(self.max_sequence_length-1):
            current_nodes = batch_data[-1].astype(int)
            node_connectivity = self.J[current_nodes, :].toarray()
            probs = softmax(node_connectivity, axis=1)
            next_nodes = [np.random.choice(self.J.shape[0], p=probs[i, :]) for i in range(self.batch_size)]
            batch_data.append(np.array(next_nodes).astype(int))
        return np.stack(batch_data, axis=1)


if __name__ == "__main__":
    from connectome_tools.connectome_loaders import load_flywire
    data_path = "../connectome_examples/data/"
    neurons, J, nts_Js = load_flywire(data_path, by_nts=True, include_spatial=True)
    sentence_gen = SentenceGenerator(J,
                                     pre_post_neurons=10,
                                     batch_size=32,
                                     max_sequence_length=50)
    sampling_start_time = time.time()
    batch_data = sentence_gen.random_walk_gen()
    sampling_end_time = time.time()
    print("Sampling time: ", sampling_end_time - sampling_start_time)
    print(batch_data)