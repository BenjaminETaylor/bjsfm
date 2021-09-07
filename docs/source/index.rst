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
>>> a_matrix = np.array(
...     [[988374.5, 316116.9, 0.],
...      [316116.9, 988374.5, 0.],
...      [0., 0., 336128.8]]
... )
>>> thickness = 0.0072*16  # laminate thickness [in]
>>> diameter = 0.25  # diameter [in]
>>> step = 0.015  # characteristic distance [in]
>>> num_pnts = 100

Max Strain Analysis
^^^^^^^^^^^^^^^^^^^^^^

Start an analysis by creating a ``MaxStrain`` object.

>>> from bjsfm.analysis import MaxStrain
>>> # create analysis with fake strain allowables
>>> analysis = MaxStrain(
...     a_matrix, thickness, diameter,
...     et={0:0.004, 90:0.004, 45:0.004, -45:0.004},
...     ec={0:0.005, 90:0.005, 45:0.005, -45:0.005},
...     es={0:0.003, 90:0.003, 45:0.003, -45:0.003},
... )

Run an analysis by calling the ``analyze`` method.

>>> bearing = [100, 0]
>>> bypass = [300, 0, 0]
>>> # w=0 for infinite plate
>>> analysis.analyze(bearing, bypass, rc=step, num=num_pnts, w=0.)
array([[2.61882825e+01, 3.79700526e+00, 9.06850925e+00, 2.34971062e+11,
        2.61882825e+01, 3.79700526e+00, 3.00577480e+01, 2.34972471e+11],
       [3.99249374e+01, 3.84765228e+00, 9.17419967e+00, 2.40684721e+01,
        1.96742170e+01, 3.84765228e+00, ...

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
