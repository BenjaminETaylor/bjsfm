# AGENTS.md

Guidance for AI agents working in **bjsfm** — a pure-Python port of the Bolted Joint Stress
Field Model (BJSFM) fortran code for analyzing bolted joints in composite airframe structures.

## Architecture (3 layers, bottom-up)

1. **`bjsfm/lekhnitskii.py`** — the math core. Implements Lekhnitskii's anisotropic
   elasticity solutions via complex stress functions. `Hole(abc.ABC)` defines the shared
   `stress()`/`displacement()` machinery and abstract `phi_1/phi_2/phi_1_prime/phi_2_prime`
   stress functions. Two concrete subclasses:
   - `UnloadedHole` — far-field bypass loads `[Nx, Ny, Nxy]` applied at infinity.
   - `LoadedHole` — cosine-distributed bearing load `p` at angle `theta`; rotates the
     material matrix into the bearing direction, solves, then rotates results back.
2. **`bjsfm/analysis.py`** — engineering API. `Analysis` composes a `LoadedHole` +
   `UnloadedHole` and **superimposes** their results (`byp_stress + brg_stress`). `MaxStrain`
   subclass adds margin-of-safety analysis per ply angle. This is the primary user entry point.
3. **`bjsfm/plotting.py`** (contour plots) and **`bjsfm/__main__.py`** (interactive CLI,
   entry point `bjsfm=bjsfm.__main__:main`).

Key data flow: loads are `bearing=[Px, Py]` (force) and `bypass=[Nx, Ny, Nxy]`
(force/unit-length). `Analysis.bearing_angle` decomposes bearing into magnitude `p` + angle
`theta`. Finite-width is modeled via the optional `w` arg (DeJong correction in `_unloaded`).

## Project-specific conventions

- **Reference every equation.** Docstrings cite source equations like `Eq. 37.5 [2]_`
  (numbered references `[1]_`–`[4]_` are in the `lekhnitskii.py` module docstring). Preserve
  this when adding/editing methods.
- **Type hints use `numpy.typing`** via the aliases in `bjsfm/_typing.py`
  (`FloatArray`, `ComplexArray`, `IntArray`, `BoolArray` = `NDArray[np.float64]` etc.;
  `ArrayLike` for array-like inputs). `numpy.typing` carries the array *dtype* but not its
  *shape* — document shapes (e.g. `3x3`, `Nx3`) in the docstrings instead. Do **not** add a
  dependency on the unmaintained `nptyping` package.
- Stress/strain rotations go through module-level `rotate_stress`, `rotate_strain`,
  `rotate_material_matrix`, `rotate_complex_parameters` — reuse these, don't inline rotation matrices.
- `MaxStrain._equalize_dicts` forces `et`/`ec`/`es` allowable dicts to share keys (fills
  missing with `np.inf`) — analysis loops rely on this invariant.
- Invariant self-checks via `numpy.testing.assert_almost_equal` are embedded in core math
  (see `rotate_material_matrix`) — keep these assertions.

## Developer workflows

- **Run tests** (unittest-based) from project root: `python -m unittest discover -s tests`
  or `python -m unittest tests.integration.test_analysis`.
- **Fortran reference tests require a compile step first.** `tests/integration/test_lekhnitskii.py`
  imports `tests.fortran.lekhnitskii_f`, built from `tests/fortran/lekhnitskii.f`. Build it
  (needs a fortran compiler, e.g. gfortran):
  ```
  cd tests/fortran && python -m numpy.f2py -c -m lekhnitskii_f lekhnitskii.f
  ```
  Python results are validated against this original fortran code (`tests/test_data.py` holds
  shared fixtures like `QUASI`, `DIAMETER`).
- **Run the CLI**: `python -m bjsfm` (interactive prompts) or the `bjsfm` console script.
- Targets Python `~=3.9`; deps: `numpy`, `matplotlib` (see `setup.py`).

## Gotchas

- `LoadedHole` assumes `Py = 0` in `equilibrium_constants` — bearing is handled by rotating
  to `theta`, not by a generic 2D solve.
- `LoadedHole.displacement` is wrapped by `@_remove_bad_displacments`, which NaNs out points
  180° behind the bearing load.
- `__main__.py` imports `from lekhnitskii import ...` (not `bjsfm.lekhnitskii`) — only works
  when run as a module; prefer `bjsfm.lekhnitskii` for new code.
</content>
</invoke>

## Development Guidelines

### Repo Priorities

1. **Correctness** — the math must be correct and well-tested; this is the primary goal.
2. **Clarity** — code should be readable and maintainable; prefer clear, explicit code over clever optimizations. Use modern software engineering practices (e.g. modularity, separation of concerns) to keep the codebase organized and understandable.
3. **Performance** — while not the main focus, avoid egregious inefficiencies; use vectorized operations where possible.
4. **Documentation** — docstrings should be comprehensive and reference source equations; maintain the existing documentation style.
5. **Testing** — maintain high test coverage, especially for the core math; use the existing unittest framework and Fortran reference tests.