Developers
==========

Developers will need to clone or download the bjsfm project from `github <https://github.com/BenjaminETaylor/bjsfm>`_.

A portion of the original BJSFM fortran code has been provided to test the core Lekhnitskii elasticity solutions.

To compile fortran code for testing, navigate to /tests/fortran in a console and run the following command to compile
the lekhnitskii.f fortran file.

``python -m numpy.f2py -c -m lekhnitskii_f lekhnitskii.f``

You may need to download a fortran compiler. For linux, GFortran will work and should be available on your distos
software center.

Reference the `numpy documentation <https://numpy.org/doc/stable/f2py/>`_ for issues.