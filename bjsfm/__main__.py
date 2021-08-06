import numpy as np
from bjsfm.analysis import MaxStrain


print('Do you want instructions? [YES/NO]')
instructions = True if input().lower() == 'yes' else False

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
    print('Input far field and bearing forces [Nx, Ny, Nxy, Px, Py]')
nx, ny, nxy, px, py = map(float, input().split(','))

if instructions:
    print('Input width (0.0 for infinite plate)')
width = float(input())

if instructions:
    print('Input bolt diameter, number of output points, step increment and number of steps desired')
diameter, num_pts, step_size, num_steps = input().split(',')
diameter, step_size = float(diameter), float(step_size)
num_pts, num_steps = int(num_pts), int(num_steps)

if 5 in output:
    analysis = MaxStrain(
        [[a11, a12, a16],[a12, a22, a26], [a16, a26, a66]], thickness, diameter,
        et=dict(zip(angles, tension)), ec=dict(zip(angles, compression)), es=dict(zip(angles, shear))
    )
else:
    analysis = MaxStrain(
        [[a11, a12, a16],[a12, a22, a26], [a16, a26, a66]], thickness, diameter
    )


def print_title(title):
    print('{:^72}'.format(title))


def print_string_fields(fields):
    assert len(fields) == 8, f'{len(fields)} fields given. must be 8.'
    for field in fields:
        print(f'{field:^9}', end='')
    print()


def print_number_fields(fields):
    assert len(fields) == 8, f'{len(fields)} fields given. must be 8.'
    for field in fields:
        print(f'{field:>9.3g}', end='')
    print()


def principal_components(xy_components):
    sx = xy_components[:, 0]
    sy = xy_components[:, 1]
    sxy = xy_components[:, 2]
    max_prin = (sx + sy)/2 + np.sqrt(((sx - sy)/2)**2 + sxy**2)
    min_prin = (sx + sy)/2 - np.sqrt(((sx - sy)/2)**2 + sxy**2)
    prin_angle = 1/2*np.arctan(2*sxy/(sx - sy))
    return max_prin, min_prin, prin_angle


def print_laminate_streses():
    print_title('LAMINATE STRESSES')
    print_string_fields(['DIST', 'ANGLE', 'X STRESS', 'Y STRESS', 'SHEAR', 'MAX.', 'MIN.', 'DIRECTION'])
    print_string_fields([' ']*4+['STRESS', 'PRINCIPAL', 'PRINCIPAL', ' '])
    for step in range(num_steps):
        distance = step*step_size
        stresses = analysis.stresses([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
        pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
        max_prin, min_prin, prin_angle = principal_components(stresses)
        for istress, stress in enumerate(stresses):
            print_number_fields([distance, pnt_angles[istress], stress[0], stress[1], stress[2], 
                max_prin[istress], min_prin[istress], prin_angle[istress]])

if 1 in output:
    print_laminate_streses()



    


