.. toctree::
   :maxdepth: 2
   :hidden:

   api
   dev

Quick Start
===========

Bolted Joint Stress Field Model (BJSFM) is a common analytical method used to analyze bolted joints in composite
airframe structures. This project ports the original fortran code to pure python code using the underlying theory.

Installation
------------

``pip install bjsfm``

Usage
-----

All examples below use the following inputs.

>>> import numpy as np
>>> # A-matrix from CLPT
>>> a = np.array(
...     [[988374.5, 316116.9, 0.],
...      [316116.9, 988374.5, 0.],
...      [0., 0., 336128.8]]
... )
>>> t = 0.0072*16  # laminate thickness [in]
>>> d = 0.25  # diameter [in]
>>> step = 0.015  # characteristic distance [in]
>>> num_pnts = 100

Max Strain Analysis
^^^^^^^^^^^^^^^^^^^^^^

Start an analysis by creating a ``MaxStrain`` object.

>>> from bjsfm.analysis import MaxStrain
>>> # create analysis with fake strain allowables
>>> analysis = MaxStrain(
...     a, t, d,
...     et0=0.004000, et90=0.004000, et45=0.004000, etn45=0.004000,
...     ec0=0.005000, ec90=0.005000, ec45=0.005000, ecn45=0.005000,
...     es0=0.003000, es90=0.003000, es45=0.003000, esn45=0.003000,
... )

Run an analysis by calling the ``analyze`` method.

>>> bearing = [100, 0]
>>> bypass = [300, 0, 0]
>>> # w=0 for infinite plate
>>> analysis.analyze(bearing, bypass, rc=step, num=num_pnts, w=0.)
array([[-8.66380409e-01, -5.87831365e-01,  2.59859196e+09,
    -6.39183215e-01, -6.39183215e-01, -9.36338750e-01],
   [-8.64977790e-01, -5.83382850e-01, -6.67315304e-01,
    -7.80050097e-01,  6.44289105e-02, -9.35666613e-01],
   ...

Lekhitskii's Solutions
^^^^^^^^^^^^^^^^^^^^^^

Create infinite plates with holes, loaded with bearing and bypass.

>>> from bjsfm.lekhnitskii import UnloadedHole, LoadedHole
>>> a_inv = np.linalg.inv(a)
>>> bypass = [100., 50., 25.]  # [Nx, Ny, Nxy] lb/in
>>> bearing = 100.  # bearing force [lb]
>>> alpha = 25.  # bearing angle [deg]
>>> byp = UnloadedHole(bypass, d, t, a_inv)
>>> brg = LoadedHole(bearing, d, t, a_inv, theta=np.deg2rad(alpha))

Obtain stresses from the plates and combine.

>>> r = np.array([d/2 + step] * num_pnts)
>>> theta = np.linspace(0, 2 * np.pi, num=num_pnts, endpoint=False)
>>> x = r * np.cos(theta)
>>> y = r * np.sin(theta)
>>> byp_stress = byp.stress(x, y)
>>> brg_stress = brg.stress(x, y)
>>> total_stress = byp_stress + brg_stress
>>> # use total stress for failure calculations ...

Plot the combined stresses.

>>> from bjsfm import plotting
>>> plotting.plot_stress(brg, byp)

.. image:: ../img/example_stress_plot.png
   :height: 350px

Contribute
----------

- Issue Tracker: https://github.com/BenjaminETaylor/bjsfm/issues
- Source Code: https://github.com/BenjaminETaylor/bjsfm

Support
-------

benjaminearltaylor@gmail.com

License
-------

This project is licensed under the MIT license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
