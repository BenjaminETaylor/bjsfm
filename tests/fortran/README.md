# Fortran reference build

`tests/integration/test_lekhnitskii.py` validates the Python port against the **original BJSFM
Fortran** routines in `lekhnitskii.f` (`UNLODED`, `LOADED`, `SIMULT`, `ROOTS`). Those routines are
exposed to Python as the compiled extension module `lekhnitskii_f` via
[`f2py`](https://numpy.org/doc/stable/f2py/).

The compiled extension is **not** checked into the repo, so you must build it once before running
the comparison tests. If it is missing, `test_lekhnitskii.py` skips (it does not error).

## Prerequisites

- A Fortran compiler (`gfortran`, typically from `gcc`).
  - macOS (Homebrew): `brew install gcc`
  - Debian/Ubuntu: `sudo apt-get install gfortran`
- `numpy` (provides `f2py`).
- On Python ≥ 3.12, `numpy.distutils` is removed, so `f2py` needs the **meson** backend:
  `pip install meson ninja` (then pass `--backend meson` to the build command below).

## Build

From the project root:

```bash
cd tests/fortran
python -m numpy.f2py -c -m lekhnitskii_f lekhnitskii.f
```

On Python ≥ 3.12:

```bash
cd tests/fortran
python -m numpy.f2py -c --backend meson -m lekhnitskii_f lekhnitskii.f
```

This produces a `lekhnitskii_f.*.so` (or `.pyd`) next to `lekhnitskii.f`. After that, run the
comparison tests from the project root:

```bash
python -m unittest tests.integration.test_lekhnitskii
```

