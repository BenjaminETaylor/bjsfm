import numpy as np
from bjsfm import analysis
from bjsfm.lekhnitskii import LoadedHole, UnloadedHole
from bjsfm.analysis import MaxStrain


print('Do you want instructions? [YES/NO]')
instructions = True if input() == 'YES' else False

if instructions:
    print('Select desired output from the following cases:\n\
        1  Laminate stresses\n\
        2  Laminate strains\n\
        3  Circumferential & radial stresses/strains\n\
        4  Displacements\n\
        5  Laminate max strain margins')
output = list(map(int, input().split(',')))

if instructions:
    print('Input A-matrix terms [A11, A12, A16, A22, A26, A66]')
a11, a12, a16, a22, a26, a66 = map(float, input().split(','))

if 5 in output:
    if instructions:
        print('Input analysis angles [<angle-1>, <angle-2>, ...]')
    angles = list(map(float, input().split(',')))
    if instructions:
        print('Input tension allowables at each angle [<tens-1>, <tens-2>, ...]')
    tension = list(map(float, input().split(',')))
    if instructions:
        print('Input compression allowables at each angle [<comp-1>, <comp-2>, ...]')
    compression = list(map(float, input().split(',')))
    if instructions:
        print('Input shear allowables at each angle [<shear-1>, <shear-2>, ...]')
    shear = list(map(float, input().split(',')))

if instructions:
    print('Input laminate thickness')
thickness = float(input())

if instructions:
    print('Input far field stresses, off-axis angle, bearing stress and bolt loading angle [Px, Py, Pxy, alpha, Pbr, theta]')
px, py, pxy, alpha, pbr, theta = list(map(float, input().split(',')))

if instructions:
    print('Input width (0.0 for infinte plate)')
width = float(input())

if instructions:
    print('Input bolt diameter, number of output points, step increment and number of steps desired')
diameter, num_pts, step, num_steps = list(map(float, input().split(',')))

if 5 in output:
    analysis = MaxStrain(
        [[a11, a12, a16],[a12, a22, a26], [a16, a26, a66]], thickness, diameter,
        et=dict(zip(angles, tension)), ec=dict(zip(angles, compression)), es=dict(zip(angles, shear))
    )
else:
    analysis = MaxStrain(
        [[a11, a12, a16],[a12, a22, a26], [a16, a26, a66]], thickness, diameter
    )

def output_streses():
    pass

    


