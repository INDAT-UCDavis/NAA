import naa
import argparse

parser = argparse.ArgumentParser(
    prog='Neutron Activation Analysis (NAA)',
    description='This program takes in a spectrum file '+
        'and applies an energy calibration from HPGe data,' +
        ' then is tries to fit isotopes to the data.',
    epilog='...'
)
parser.add_argument(
    'input_file', metavar='<str>.TKA', type=str,
    help='input file specification NAA analysis'
)
parser.add_argument(
    '-calibration', dest='calibration', default=naa.calibration_2022,
    help='HPGe calibration function to use.'
)
parser.add_argument(
    '-tolerance', dest='tolerance', default=5.0,
    help='tolerance of the peak fitting.'
)

if __name__ == "__main__":
    
    args = parser.parse_args()
    naa_results = naa.fit_spectrum(
        args.input_file,
        args.calibration,
        args.tolerance
    )