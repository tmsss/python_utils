import numpy as np
import scipy as sp

def find_common(arrays_):
    '''
    Take a list of arrays and return the common elements between them
    '''
    return set.intersection(*[set(list) for list in arrays_])


def find_max(*args):
    all = sum(args, [])
    return max(all)


# flatten/merge a list of arrays
def flatten_list(arrays_):
    return [item for sublist in arrays_ for item in sublist]


# Jensen-Shannon Divergence from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence#27432724
# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
def get_jsd(p, q, base=2):
    # https://gist.github.com/zhiyzuo/f80e2b1cfb493a5711330d271a228a3d

    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)

    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)

    jsd = sp.stats.entropy(p,m, base=base)/2. + sp.stats.entropy(q, m, base=base)/2.

    # print(jsd)

    return jsd



# calculate cosine similarity
def get_cosim(corpus_01, corpus_02):
    return np.dot(corpus_01, corpus_02)/(np.linalg.norm(corpus_01)*np.linalg.norm(corpus_02))


# calculate euclidean distance
def get_ecd(corpus_01, corpus_02):
    return np.linalg.norm(corpus_01 - corpus_02)
