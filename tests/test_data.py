import numpy as np


####################################################################################################################
# geometry inputs
####################################################################################################################
DIAMETER = 0.15  # hole diameter
STEP_DIST = 0.25  # distance to test away from hole edge (must be >0 for tests to pass)

####################################################################################################################
# test coverage inputs
####################################################################################################################
NUM_POINTS = 500  # number of points to test around circumference

####################################################################################################################
# material inputs
####################################################################################################################
# Hexcel 8552 IM7 Unidirectional
# ref: https://www.wichita.edu/research/NIAR/Documents/Qual-CAM-RP-2009-015-Rev-B-Hexcel-8552-IM7-MPDR-04.16.19.pdf
# E1c[RTD] = 20.04 Msi
# E2c[RTD] = 1.41 Msi
# nu12[RTD] = 0.356
# nu21[RTD] = 0.024
# G12[RTD] = 0.68 Msi
# CPT = 0.0072 in
# QUASI [25/50/25], [45/0/-45/90]2s
# HARD [50/40/10], [0/45/0/90/0/-45/0/45/0/-45]s
# SOFT [10/80/10], [45/-45/0/45/-45/90/45/-45/45/-45]s
QUASI = np.array(
    [[988374.5, 316116.9, 0.],
     [316116.9, 988374.5, 0.],
     [0., 0., 336128.8]]
)
HARD = np.array(
    [[1841084.0, 330697.9, 0.],
     [330697.9, 758748.5, 0.],
     [0., 0., 355712.8]]
)
SOFT = np.array(
    [[1042123.5, 588490.7, 0.],
     [588490.7, 1042123.5, 0.],
     [0., 0., 613505.6]]
)

QUASI_INV = np.linalg.inv(QUASI)
HARD_INV = np.linalg.inv(HARD)
SOFT_INV = np.linalg.inv(SOFT)

QUASI_THICK = 0.0072*16  # 0.1152
HARD_THICK = 0.0072*20  # 0.144
SOFT_THICK = 0.0072*20  # 0.144

E_QUASI = 1/(QUASI_INV[0, 0]*QUASI_THICK)
G_QUASI = 1/(QUASI_INV[2, 2]*QUASI_THICK)
E_HARD = 1/(HARD_INV[0, 0]*HARD_THICK)
G_HARD = 1/(HARD_INV[2, 2]*HARD_THICK)
E_SOFT = 1/(SOFT_INV[0, 0]*SOFT_THICK)
G_SOFT = 1/(SOFT_INV[2, 2]*SOFT_THICK)

####################################################################################################################
# strength inputs & strain allowables
####################################################################################################################
# Hexcel 8552 IM7 Unidirectional
# ref: https://www.wichita.edu/research/NIAR/Documents/NCP-RP-2009-028-Rev-B-HEXCEL-8552-IM7-Uni-SAR-4-16-2019.pdf
# mean values (minimum of ETW, RTD and CTD where available)
SHEAR_STRN = 16.56e3/0.68e6
QUASI_UNT = 99.35e3/E_QUASI
QUASI_UNC = 57.68e3/E_QUASI

HARD_UNT = 174.18/E_HARD
HARD_UNC = 79.42/E_HARD

SOFT_UNT = 54.17/E_HARD
SOFT_UNC = 40.61/E_HARD

####################################################################################################################
# test point inputs
####################################################################################################################
# to match the original BJSFM output, points to test must be equally spaced around hole, starting at zero degrees
# there must be two rows of points; one at the hole boundary, and another at step distance
X_POINTS = [r * np.cos(theta) for r, theta in
            zip([DIAMETER / 2] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
X_POINTS += [r * np.cos(theta) for r, theta in
             zip([DIAMETER / 2 + STEP_DIST] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
Y_POINTS = [r * np.sin(theta) for r, theta in
            zip([DIAMETER / 2] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]
Y_POINTS += [r * np.sin(theta) for r, theta in
             zip([DIAMETER / 2 + STEP_DIST] * NUM_POINTS, np.linspace(0, 2 * np.pi, num=NUM_POINTS, endpoint=False))]




