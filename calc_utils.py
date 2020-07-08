import numpy as np
import scipy as sp
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
from scipy.stats import entropy
import ast
from tqdm import tqdm
import multiprocessing as mp
import time
import itertools
import powerlaw
import matplotlib.pyplot as plt


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

# convert list string literal to list
def str_list(lst):
    try:
        lst = lst.replace('][', ',')
        return np.array(ast.literal_eval(lst))
    except Exception:
        print(lst)
        pass


# apply function to list of arguments
def apply_fn(fn, args):
    for item in tqdm(args):
        fn(*item)


# iterate list in chunks
# from https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def get_square_matrix(p, q):
    """Return arrays with the size of the smaller array between p and q.

    Input
    -----
    p, q : arrays

    Returns
    -----
    p, q : arrays with the size of the smaller array between p and q.
    """
    
    minimun = min([len(p), len(q)])

    return p[:minimun], q[:minimun]


def get_jsd(p, q, dist=False):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    Jensen-Shannon Divergence from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence#27432724
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Input
    -----
    P, Q : array-like probability distributions of equal length that sum to 1

    distance: the square root of the divergence
    """
    # make sure if the arrays are of equal size
    p, q = get_square_matrix(p, q)

    # convert to np.array and flatten
    p, q = np.asarray(p).flatten(), np.asarray(q).flatten()

    jsd = distance.jensenshannon(p, q)

    if dist:
        return jsd
    else:
        return jsd ** 2


def rle(series):
    """ 
    Run length encoding for sparse time series to encode zeros as in needed for awarp calculation 
    (https://ieeexplore.ieee.org/document/7837859 | https://github.com/mclmza/AWarp)
    
    args
    ----
    series: sparse times series (e.g. x = [0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 4, 0, 0])

    returns
    ---- 
    array with encoded zeros (e.g. [3 2 3 1 5 6 2 4 2]) """

    # convert to np array
    series = np.array(series)

    # add points to detect inflection on start and end
    series_ = np.concatenate(([1], series, [1]))

    # find zeros and non zeros
    zeros = np.where(series_ == 0)[0]

    if len(zeros) > 0:
        nonzeros = np.where(series_ != 0)[0]

        # detect zero sequencies
        split_zeros = np.where(np.diff(zeros) > 1)[0] + 1

        splitted_zeros = np.split(zeros, split_zeros)

        zero_points = []
        zero_points = np.array(zero_points, dtype=int)
        
        for z in splitted_zeros:
            zero_points = np.append(zero_points, z[-1])

        # detect non-zero sequencies
        nonzero_points = nonzeros[np.where(np.diff(nonzeros) > 1)[0]]

        # concat all splitting points
        split = np.sort(np.concatenate([zero_points, nonzero_points]))

        # avoid splitting on first element
        split = split[split > 0]

        # separate zero sequencies from non-zero sequencies
        splitted_series = np.split(series, split)

        # initialize empty array
        rle = []
        rle = np.array(rle, dtype=int)

        # encode zeros
        for s in splitted_series:
            # if it is a zero sequence enconde the lenght of the sequence
            if np.sum(s) == 0:
                rle = np.append(rle, [len(s)], axis=0)
            else:
                rle = np.concatenate([rle, s])

        # remove zeros in the end
        rle = rle[rle > 0]

        return rle
    else:
        return series

# x = [1, 2, 2, 3, 0, 0, 0, 5, 0, 6, 4]
# print(rle(x))


# calculate euclidean distance
def get_ecd(p, q, square=True):

    # make sure if the arrays are of equal size
    if square:
        p, q = get_square_matrix(p, q)

    return np.linalg.norm(p - q)


# calculate cosine similarity
def get_cosim(corpus_01, corpus_02):
    return np.dot(corpus_01, corpus_02)/(np.linalg.norm(corpus_01)*np.linalg.norm(corpus_02))


def itertools_flatten(arr_):
    return list(itertools.chain.from_iterable(arr_))


def get_flatten(arr_, size):
    """Parallel function to flatten a N dimensional array/matrix in chunks.

    Input
    -----
    arr_ : N Dimensional array

    size: int to define the number of array items in each chunk 

    Returns
    -----
    flatten_ : 1D array
    """

    chunks = []

    for chunk in chunker(arr_, size):
        # with mp.get_context("spawn").Pool(processes=int(mp.cpu_count())) as pool:
        #     chunk_data = pool.starmap(itertools_flatten, chunk)
        #     pool.close()
        #     pool.join()
        chunks.append(itertools_flatten(chunk))

    # print(chunks)
    flatten_ = itertools_flatten(chunks)
    # print(len(flatten_))

    return flatten_


def get_entropy(corpus):
    """
    Computes entropy for a given array

    Parameters
    ----------
    corpus : 1D array

    Returns
    -------
    entropy: float
        the computed entropy
    """

    # make sure that we dealing with an 1D array
    # corpus = np.asarray(corpus).flatten()
    # ray.init()

    len_corpus = corpus.shape[0] * corpus.shape[1]

    flatten_corpus = get_flatten(corpus, 50)

    assert len_corpus == len(flatten_corpus), "The length of the flattened array ({}) doesn't correspond to expected ({})".format(len(flatten_corpus), len_corpus)

    t = time.process_time()

    result = entropy(flatten_corpus)

    # ray.shutdown()

    print('Elapsed time: {}'.format(time.process_time() - t))
    print('Entropy calculation finished')
    return result


def get_powerlaw(distribution:list):

    # powerlaw.plot_pdf(data, color='r', ax=figPDF)
    # fit.plot_pdf(label="Fitted PDF")
    # figPDF.set_ylabel(r"$p(X)$")
    # figPDF.set_xlabel(r"Word Frequency")
    fit = powerlaw.Fit(distribution, discrete=True)
    print('alpha: {}'.format(fit.power_law.alpha))
    print('min: {}'.format(fit.power_law.xmin))
    # powerlaw.plot_pdf(data[data>=fit.power_law.xmin], label="Data as PDF")
        
    R, p = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
    print(R, p)
    print('power law  parameter (alpha): {}'.format(fit.truncated_power_law.parameter1))
    print('exponential cut-off parameter (beta): {}'.format(fit.truncated_power_law.parameter2))

    # figPDF = powerlaw.plot_pdf(distribution, color='b')
    # powerlaw.plot_pdf(distribution, color='r', ax=figPDF)
    # fit.power_law.plot_pdf(label="Fitted PDF", ls=":")
    # plt.legend(loc=3, fontsize=14)

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
