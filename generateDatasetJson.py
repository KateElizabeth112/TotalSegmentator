# Generate the dataset.json file
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import argparse
import os

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--root_dir", default='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet', help="Root directory for nnUNet")
parser.add_argument("-n", "--dataset_name", default='Dataset501_Set1', help="Name of the dataset")
parser.add_argument("-tc", "--training_cases", default=128)
args = vars(parser.parse_args())

# set up variables
ROOT_DIR = args['root_dir']
DS_NAME = args['dataset_name']
TC = args['training_cases']

# print
print("Generating dataset.json....")
print("Root directory: {}".format(ROOT_DIR))
print("Dataset name: {}".format(DS_NAME))
print("Training cases: {}".format(TC))

output_dir = os.path.join(ROOT_DIR, "nnUNet_raw/{}".format(DS_NAME))
imagesTr_dir = os.path.join(ROOT_DIR, "nnUNet_raw/{}/imagesTr".format(DS_NAME))

channel_names = {0: "CT"}

labels = {"background": 0,
          "right kidney": 1,
          "left kidney": 2,
          "liver": 3,
          "pancreas": 4}

file_ending = ".nii.gz"

generate_dataset_json(str(output_dir),
                      channel_names,
                      labels,
                      int(TC),
                      file_ending)

print("Finished")