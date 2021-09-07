import sys, re
import numpy as np
from bjsfm.analysis import MaxStrain
from lekhnitskii import rotate_stress, rotate_strain


def check_input(regex: str, error_msg: str = 'Please try again...'):
    while True:
        attempt = input()
        if attempt == '':
            print(error_msg)
            continue
        if str.lower(attempt) in ('exit', 'quit'):
            sys.exit('Program terminated.')
        match = re.fullmatch(regex, attempt)
        if match:
            return attempt
        print(error_msg)


def main():
    """This is the main entry point"""

    # regex for testing input
    numeric = r'[0-9]*\.?[0-9]*(, ?[0-9]*\.?[0-9]*)*'

    print('Do you want instructions? [YES/NO]')
    instructions = True if check_input(r'\w{2,3}').lower() == 'yes' else False

    if instructions:
        print('Select desired output from the following cases:\n\
            1  Laminate stresses\n\
            2  Laminate strains\n\
            3  Circumferential & radial stresses/strains\n\
            4  Displacements\n\
            5  Laminate max strain margins')
    output = list(map(int, check_input(numeric).split(',')))

    if instructions:
        print('Input A-matrix terms [A11, A12, A16, A22, A26, A66]')
    a11, a12, a16, a22, a26, a66 = map(float, check_input(numeric).split(','))

    if 5 in output:
        if instructions:
            print('Input analysis angles [<angle-1>, <angle-2>, ...]')
        angles = list(map(int, check_input(numeric).split(',')))
        if instructions:
            print('Input tension strain allowables at each angle [<tens-1>, <tens-2>, ...]')
        tension = list(map(float, check_input(numeric).split(',')))
        if instructions:
            print('Input compression strain allowables at each angle [<comp-1>, <comp-2>, ...]')
        compression = list(map(float, check_input(numeric).split(',')))
        if instructions:
            print('Input shear strain allowables at each angle [<shear-1>, <shear-2>, ...]')
        shear = list(map(float, check_input(numeric).split(',')))

    if instructions:
        print('Input laminate thickness')
    thickness = float(check_input(numeric))

    if instructions:
        print('Input far field and bearing forces [Nx, Ny, Nxy, Px, Py]')
    nx, ny, nxy, px, py = map(float, check_input(numeric).split(','))

    if instructions:
        print('Input width (0.0 for infinite plate)')
    width = float(check_input(numeric))

    if instructions:
        print('Input bolt diameter, number of output points, step increment and number of steps desired')
    diameter, num_pts, step_size, num_steps = check_input(numeric).split(',')
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
        # assert len(fields) == 8, f'{len(fields)} fields given. must be 8.'
        for field in fields:
            print(f'{field:^9}', end='')
        print()


    def print_number_fields(fields):
        # assert len(fields) == 8, f'{len(fields)} fields given. must be 8.'
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
        print_string_fields(['DIST', 'ANGLE', 'X STRESS', 'Y STRESS', 'SHEAR', 'MAX', 'MIN', 'DIRECTION'])
        print_string_fields([' ']*4+['STRESS', 'PRINCIPAL', 'PRINCIPAL', ' '])
        for step in range(num_steps):
            distance = step*step_size
            stresses = analysis.stresses([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
            max_prin, min_prin, prin_angle = principal_components(stresses)
            for istress, stress in enumerate(stresses):
                print_number_fields([distance, pnt_angles[istress], stress[0], stress[1], stress[2], 
                    max_prin[istress], min_prin[istress], prin_angle[istress]])
        print()


    def print_displacements():
        print_title('DISPLACEMENTS')
        print_string_fields(['DIST', 'ANGLE', 'U', 'V'])
        for step in range(num_steps):
            distance = step*step_size
            displacements = analysis.displacements([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
            for idisp, disp in enumerate(displacements):
                print_number_fields([distance, pnt_angles[idisp], disp[0], disp[1]]) 
        print()


    def print_laminate_strains():
        print_title('LAMINATE STRAINS')
        print_string_fields(['DIST', 'ANGLE', 'X STRAIN', 'Y STRAIN', 'SHEAR', 'MAX', 'MIN', 'DIRECTION'])
        print_string_fields([' ']*4+['STRAIN', 'PRINCIPAL', 'PRINCIPAL', ' '])
        for step in range(num_steps):
            distance = step*step_size
            strains = analysis.strains([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
            max_prin, min_prin, prin_angle = principal_components(strains)
            for istrain, strain in enumerate(strains):
                print_number_fields([distance, pnt_angles[istrain], strain[0], strain[1], strain[2], 
                    max_prin[istrain], min_prin[istrain], prin_angle[istrain]])
        print()


    def print_circumferential_radial():
        print_title('CIRCUMFERENTIAL AND RADIAL STRESSES & STRAINS')
        print_string_fields(['DIST', 'ANGLE', 'THETA', 'RADIAL', 'SHEAR', 'THETA', 'RADIAL', 'SHEAR'])
        print_string_fields([' ']*2 + ['STRESS']*3 + ['STRAIN']*3)
        for step in range(num_steps):
            distance = step*step_size
            strains = analysis.strains([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            stresses = analysis.stresses([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
            for iangle, pnt_angle in enumerate(pnt_angles):
                strain = rotate_strain(strains[iangle], np.deg2rad(pnt_angle))
                stress = rotate_stress(stresses[iangle], np.deg2rad(pnt_angle))
                print_number_fields([distance, pnt_angle, stress[0], stress[1], stress[2], 
                    strain[0], strain[1], strain[2]]) 
        print()


    def print_margins():
        print_title('MAX STRAIN MARGINS')
        print_string_fields(['DIST', 'ANGLE'] + 
            [f(angle) for angle in analysis.angles 
                for f in (lambda angle: f'{angle}AXIAL', lambda angle: f'{angle}SHEAR')])
        for step in range(num_steps):
            distance = step*step_size
            pnt_angles = np.linspace(0, 360, num_pts, endpoint=False)
            margins = analysis.analyze([px, py], [nx, ny, nxy], rc=distance, num=num_pts, w=width)
            for iangle, angle in enumerate(pnt_angles):
                print_number_fields([distance, angle] + margins[iangle].tolist()) 
        print()


    if 1 in output:
        print_laminate_streses()

    if 2 in output:
        print_laminate_strains()

    if 3 in output:
        print_circumferential_radial()

    if 4 in output:
        print_displacements()

    if 5 in output:
        print_margins()
    
    return 0  # successful



if __name__ == '__main__':
    sys.exit(main())