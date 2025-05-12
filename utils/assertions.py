from numpy import ndarray

def assert_same_shape(first: ndarray,
                      second: ndarray):
    assert first.shape == second.shape, \
        '''
        Two tensors should have the same shape, got {0} and {1} instead.
        '''.format(tuple(first.shape), tuple(second.shape))
    return None