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


# Jensen-Shannon Divergence from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence#27432724
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

# From the original example with three distributions:
# P_1 = np.array([1/2, 1/2, 0])
# P_2 = np.array([0, 1/10, 9/10])
# P_3 = np.array([1/3, 1/3, 1/3])
#
# prob_distributions = np.array([P_1, P_2, P_3])
# n = len(prob_distributions)
# weights = np.empty(n)
# weights.fill(1/n)
#
# print(JSD(prob_distributions, weights))
