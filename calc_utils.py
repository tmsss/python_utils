import numpy as np
import scipy as sp
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

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


# build list from dict
def build_list_dict(array_, field_, list_):
    return [array_.append(tag[field_]) or tag[field_]
            for tag in list_]


def build_list(array_, list_):
    return [array_.append(ix) or ix for ix in list_]


# remove one list of items from other
def remove_list(all_items, list_):
    return [item for item in all_items if item not in list_]


# flatten json files
def get_values(lVals):
    res = []
    for val in lVals:
        if type(val) not in [list, set, tuple]:
            res.append(val)
        else:
            res.extend(get_values(val))
    return res


# apply function to list of arguments
def apply_fn(fn, args):
    for item in tqdm(args):
        fn(*item)


# iterate list in chunks
# from https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


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

    return jsd


# calculate cosine similarity
def get_cosim(corpus_01, corpus_02):
    return np.dot(corpus_01, corpus_02)/(np.linalg.norm(corpus_01)*np.linalg.norm(corpus_02))


# calculate euclidean distance
def get_ecd(corpus_01, corpus_02):
    return np.linalg.norm(corpus_01 - corpus_02)


# linkage function to create dendograms
'''
from https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

method=’single’ assigns for all points in cluster and in cluster. This is also known as the Nearest Point Algorithm.

method=’complete’ assigns for all points in cluster u and in cluster . This is also known by the Farthest Point Algorithm or Voor Hees Algorithm.

method=’average’ assigns for all points and where and are the cardinalities of clusters and, respectively. This is also called the UPGMA algorithm.

method=’weighted’ assigns where cluster u was formed with cluster s and t and v is a remaining cluster in the forest. (also called WPGMA)

method=’centroid’ assigns where and are the centroids of clusters and , respectively. When two clusters and are combined into a new cluster , the new centroid is computed over all the original objects in clusters and . The distance then becomes the Euclidean distance between the centroid of and the centroid of a remaining cluster in the forest. This is also known as the UPGMC algorithm.

method=’median’ assigns like the centroid method. When two clusters and are combined into a new cluster , the average of centroids s and t give the new centroid. This is also known as the WPGMC algorithm.

method=’ward’ uses the Ward variance minimization algorithm. The new entry is computed as follows, where
is the newly joined cluster consisting of clusters and , is an unused cluster in the forest, and is the cardinality of its argument. This is also known as the incremental algorithm.
'''

def get_linkage(data, method, metric):
    return linkage(data, method=method, metric=metric)
