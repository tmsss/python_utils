import numpy as np
from scipy.stats import entropy as H

def find_common(arrays_):
    '''
    Take a list of arrays and return the common elements between them
    '''
    return set.intersection(*[set(list) for list in arrays_])


def find_max(*args):
    all = sum(args, [])
    return max(all)

# function to calculate probability distributions and weights to calculate the JS divergence from an array of distributions
def get_prob_dist(dists):
    prob_distributions = np.array(dists)
    n = len(prob_distributions)
    weights = np.empty(n)
    return prob_distributions, weights.fill(1/n)


# Jensen-Shannon Divergence from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence#27432724
# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
def JSD(prob_distributions, weights, logbase=2):
    # left term: entropy of misture
    wprobs = weights * prob_distributions
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    # right term: sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_distributions])
    wentropies = weights * entropies
    sum_of_entropies = wentropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return(divergence)
