# Partridge–Schwenke water

Native potential with a McUtils `SharedLibrary`/ctypes interface.

| Interface property | Convention |
| --- | --- |
| Coordinate units | bohr |
| Output units | Hartree |
| Atom order | H,H,O |
| Input shape | `(..., 3, 3)` |

## Build and load

From this repository root, with a native compiler available:

```sh
make -j4 -C WaterPS/WaterPS/libs/PatrickShingle PYTHON=/path/to/python
```

The build uses GNU Fortran. `FC=/path/to/gfortran` selects the compiler. All units
retain `-frecursive`, which keeps local workspace independent across calls.

Use this repository on `PYTHONPATH` alongside McUtils, then:

```python
from WaterPS import load_potential
potential = load_potential()
energy = potential.get_pot(coords=coords)
```

`load_potential(library=...)` accepts an explicit shared-library path. Loading and
symbol resolution finish before it returns. Inputs are checked, batch buffers
are allocated per call, and no evaluation changes the process working directory.
The same object can be shared across a `ThreadPoolExecutor`.

## Reentrancy

The raw DATA tables remain unchanged. Each evaluation copies the immutable
parameters before applying the original scaling and unit conversions; working
arrays and scaled coefficients are local. This removes both first-call races
and the blanket SAVE of numerical workspace. The energy reference is unchanged.

## Validation

After building, with the repository on `PYTHONPATH`:

```sh
/path/to/python -m unittest discover -s tests -v
```

Tests include independent serial/reference values, scalar and batch layouts,
concurrent first evaluations, 2,048 calls across eight threads, and input/output
ownership. Where analytic gradients are available, finite differences check their
units and layout. `POTENTIAL_LIBRARY=/absolute/path/to/library.so` selects a
separate diagnostic build for the test suite.

`tests/native_checks.f90` enables IEEE invalid-operation, division-by-zero, and
overflow traps. Link it to the diagnostic object files built with bounds checks
and signaling-NaN initialization to exercise the native calculation directly.
