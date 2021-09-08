![bjsfm](https://raw.githubusercontent.com/BenjaminETaylor/bjsfm/master/docs/img/logo_02.png)
Bolted Joint Stress Field Model (BJSFM) is a common analytical method used to analyze bolted joints in composite
airframe structures. This project ports the original fortran code to pure python code using the underlying theory.

```
    from bjsfm.analysis import MaxStrain
    a_matrix = [[988374.5, 316116.9, 0.],
                [316116.9, 988374.5, 0.],
                [0., 0., 336128.8]]
    thickness = 0.1152
    diameter = 0.25
    analysis = MaxStrain(a_matrix, thickness, diameter)

    # get stresses, strains and displacements at four points around hole
    bearing = [100, 0]  #[Px, Py]
    bypass = [100, 0, 0]  #[Nx, Ny, Nxy]
    analysis.stresses(bearing, bypass, num=4)
    analysis.strains(bearing, bypass, num=4)
    analysis.displacements(bearing, bypass, num=4)

    # plot stresses
    analysis.plot_stress(bearing, bypass)
```

## Installation

`pip install bjsfm`

## Documentation

https://bjsfm.readthedocs.io

## Features

- [X] Lekhnitskii's anisotropic elasticity solutions for loaded (cosine distribution) and unloaded holes
    - [X] stresses
    - [X] displacements
- [X] Combined bearing and bypass 2D infinite plate stress distribution
- [X] Optional DeJong tension (or compression) finite width correction
- [X] Max strain analysis
- [ ] Plotting
    - [X] stresses
    - [ ] displacements
- [X] Command-line Interface (CLI)

## Contribute

- Issue Tracker: https://github.com/BenjaminETaylor/bjsfm/issues
- Source Code: https://github.com/BenjaminETaylor/bjsfm

## Support

benjaminearltaylor@gmail.com

## License

This project is licensed under the MIT license.