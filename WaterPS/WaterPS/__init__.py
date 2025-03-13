
import os, sys, numpy as np

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libs"))
    from PatrickShingle import *
finally:
    sys.path.pop(0)

def _potential(coords, atoms, *args):
    coords = np.ascontiguousarray(coords)
    if coords.ndim > 2:
        main_shape = coords.shape[:-2]
        num_walkers = int(np.product(main_shape))
        coords = coords.reshape((num_walkers,) + coords.shape[-2:])
        res = calc_hoh_pot(coords, len(coords)) # np.array([getpot(c)[0] for c in coords])
        res = res.reshape(main_shape)
    else:
        res = calc_hoh_pot(coords[np.newaxis], 1)
    return res
