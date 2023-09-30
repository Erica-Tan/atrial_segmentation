import os
import json

from utils.utils import get_image_files


base_dir = "./data/"
data_dir = os.path.join(base_dir, "dataset_lascar_processed")

patients = os.listdir(data_dir)

# training 30, testing 11
n_test_patients = 10
n_validate_patients = 10
length = len(patients)
train_patient_list, val_patient_list, test_patient_list = (patients[:-(n_validate_patients+n_test_patients)], \
                                                          patients[-(n_validate_patients+n_test_patients):-n_test_patients], \
                                                          patients[-n_test_patients:])


train_files = get_image_files(data_dir, train_patient_list, has_section=True)
val_files = get_image_files(data_dir, val_patient_list, has_section=True)
test_files = get_image_files(data_dir, test_patient_list, has_section=True)


json_files = {
    "description": "LAScarQS2022 dataset",
    "modality": {
        "0": "LGE-MRIs"
    },
    "labels": {
        "0": "background",
        "1": "LA endo",
    },

    "name": "LAScarQS2022",
    "numTest": len(test_files),
    "numTraining": len(train_files),
    "tensorImageSize": "3D",
    "training": train_files,
    "validation": val_files,
    "testing": test_files
}


final = json.dumps(json_files)

with open(os.path.join(base_dir, "dataset_lascar.json"), "w") as outfile:
    outfile.write(final)