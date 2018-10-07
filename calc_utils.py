def find_common(arrays_):
    '''
    Take a list of arrays and return the common elements between them
    '''
    return set.intersection(*[set(list) for list in arrays_])


def find_max(*args):
    all = sum(args, [])
    return max(all)
