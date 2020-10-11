![bjsfm](https://raw.githubusercontent.com/BenjaminETaylor/bjsfm/master/docs/img/logo_02.png)
Bolted Joint Stress Field Model (BJSFM) is a common analytical method used to analyze bolted joints in composite
airframe structures. This project ports the original fortran code to pure python code using the underlying theory.

```
    from bjsfm.lekhnitskii import UnloadedHole
    a_inv = [[0.1, 0.05, 0.], [0.05, 0.1, 0.], [0., 0., 0.5]]  # inverse a-matrix from CLPT
    loads = [100, 100, 50]  # force / unit length
    plate = UnloadedHole(diameter=0.25, thickness=0.1, a_inv=a_inv, loads=loads)

    # get stresses at four points around hole
    plate.stress(x=[0.125, 0., -0.125, 0.], y=[0., 0.125, 0., -0.125])
    
    # plot stresses
    from bjsfm import plotting
    plotting.plot_stress(plate)
```

## Installation

`pip install bjsfm`

## Documentation

https://bjsfm.readthedocs.io

## Features

- [ ] Lekhnitskii's anisotropic elasticity solutions for loaded (cosine distribution) and unloaded holes
    - [X] stresses
    - [ ] displacements
- [ ] Combined bearing and bypass 2D infinite plate stress distribution
- [ ] Optional DeJong tension (or compression) bearing correction

## Contribute

- Issue Tracker: https://github.com/BenjaminETaylor/bjsfm/issues
- Source Code: https://github.com/BenjaminETaylor/bjsfm

## Support

benjaminearltaylor@gmail.com

## License

This project is licensed under the MIT license.