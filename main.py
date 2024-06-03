import jax.numpy as jnp
import numpy as np
import jax

import argparse

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_neurons', default=139255, type=int, help="# of neurons")
    parser.add_argument('--dim', default=139255, type=int, help="Word2vec dimension")
    parser.add_argument('--window_size', default=2, type=int, help="Window size +/-, so it's 2*window_size + 1")
    parser.add_argument('--init_seed', default=1, type=int, help="Init seed")

    parser.add_argument('--steps', default=1, type=int, help='Number of steps to run')
    parser.add_argument('--ckpt_every', default=1, type=int, help='How often to dump params')

def dummy_data_generator(batch_size=128, seq_len=64, minval=0, maxval=MAX_NEURONS, key=jax.random.PRNGKey(0)):
    key, this_key = jax.split(key, 2)
    yield jax.random.randint(key, shape=(batch_size, seq_len), minval, maxval)


def single_word_loss(params, window):
    '''
    Assumes center of window is being predicted from context, ordered
    '''
    half = window.shape[0]//2
    target_word = window[half]
    input_words = jnp.concatenate([window[:half], window[half+1:]])
    logits = jnp.einsum('ij,i->j', params['u'], params['v'][input_words].reshape(-1))
    return -jax.nn.log_softmax(logits)[target_word]


def forward(params, sequence):
    total_ctx = 1 + params['u'].shape[0]/params['v'].shape[1]
    window_inds = jnp.arange(sequence.shape[0]-total_ctx+1, total_ctx)[:, None] + jnp.arange(total_ctx)[None, :]
    windows = sequence[window_inds]
    return jnp.mean(jax.vmap(partial(single_word_loss, params=params))(windows))


def loss(params, batch):
    return jnp.mean(jax.vmap(partial(forward, params=params))(batch))


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    v_key, window_key, u_key = jax.random.split(jax.random.PRNGKey(opts.init_seed), 3)
    fly2vec_params = {'v': jnp.random.uniform(v_key, -1/np.sqrt(opts.max_neurons), 1/np.sqrt(opts.max_neurons), shape=(opts.max_neurons, opts.dim)),
                        'u': jax.random.uniform(window_key, -1/np.sqrt(2*opts.window_size*optx.dim), 1/np.sqrt(2*opts.window_size*optx.dim), 
                                                        shape=(2*opts.window_size*opts.dim, opts.max_neurons))}

