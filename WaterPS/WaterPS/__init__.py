"""Legacy potential entry using the reentrant ctypes library."""
import numpy as np
from ..potential import load_potential
_pes = load_potential()

def _potential(coords, atoms, *args):
    return np.atleast_1d(_pes.get_pot(coords=coords))
