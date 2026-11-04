"""McUtils ctypes interface: H,H,O coordinates in bohr; energies in Hartree."""
import ctypes
from pathlib import Path
import numpy as np
from McUtils.Extensions import SharedLibrary

__all__ = ['load_potential']


def _prepare(kw):
    coords = np.asarray(kw['coords'], dtype=np.float64)
    if coords.ndim < 2 or coords.shape[-2:] != (3,3):
        raise ValueError('coords must have shape (..., 3, 3), with atoms H,H,O')
    if not np.isfinite(coords).all():
        raise ValueError('coords must be finite')
    for i,j in [(0,1),(0,2),(1,2)]:
        if np.any(np.sum((coords[...,i,:]-coords[...,j,:])**2,axis=-1)==0):
            raise ValueError('atoms must have distinct positions')
    count=coords.size//9
    if count > np.iinfo(np.int32).max:
        raise ValueError('too many geometries')
    kw['coords']=np.ascontiguousarray(coords)
    kw['ngeom']=count
    kw['energy']=np.empty(coords.shape[:-2] or (1,), dtype=np.float64)
    return kw


def _result(result, kw):
    return float(kw['energy'][0]) if kw['coords'].ndim==2 else kw['energy']


def load_potential(library=None):
    if library is None:
        library=Path(__file__).parent/'WaterPS/libs/PatrickShingle/libWaterPS.so'
    handle=ctypes.CDLL(str(Path(library).resolve()))
    potential=SharedLibrary(handle, get_pot=dict(
        name='waterps_potential', coords=[float], ngeom=int, energy=[float],
        return_type=None, defaults=dict(ngeom=0,energy=None),
        prep_args=_prepare, return_handler=_result))
    potential.get_pot.initialize()
    return potential
