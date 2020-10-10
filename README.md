![logo image](https://github.com/BenjaminETaylor/bjsfm/blob/master/docs/img/logo_02.png)
# bjsfm
Bolted Joint Stress Field Model (BJSFM) is a common analytical method used to analyze bolted joints in composite
airframe structures. This project ports the original fortran code to pure python code using the underlying theory.

```
    from bjsfm.lekhnitskii import UnloadedHole
    a_inv = [[0.1, 0.05, 0.], [0.05, 0.1, 0.], [0., 0., 0.5]]  # inverse a-matrix from CLPT
    loads = [100, 100, 50]  # force / unit length
    plate = UnloadedHole(diameter=0.25, thickness=0.1, a_inv=a_inv, loads=loads)
    # get stresses at four points around hole
    plate.stress(x=[0.125, 0., -0.125, 0.], y=[0., 0.125, 0., -0.125])
```

## Features

- [X] Lekhnitskii's anisotropic elasticity solutions for loaded (cosine distribution) and unloaded holes
- [ ] Combined bearing and bypass 2D infinite plate stress distribution
- [ ] Optional DeJong tension (or compression) bearing correction

## Installation

`pip install bjsfm`

## Contribute

- Issue Tracker: https://github.com/BenjaminETaylor/bjsfm/issues
- Source Code: https://github.com/BenjaminETaylor/bjsfm

## Support

benjaminearltaylor@gmail.com

## License

This project is licensed under the MIT license.