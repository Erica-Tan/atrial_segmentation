# Automated atrial segmentation of 3D cardiac LGE-MRIs
This repository contains the implementation of diferent architectures for atrial Segmentation


### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

## Dataset


The training data is from the [LAScarQS 2022 dataset]https://zmiclab.github.io/projects/lascarqs22/data.html . Datasets used are 3D cardiac LGE-MRIs from 3 centers,
- Target: 1. LA scar, 2. LA endo.
- Task: Segmentation
- Modality: LGE-MRI
- Size: 3D volumes


We provide the json file that is used to train our models in the following link:

https://github.com/Erica-Tan/atrial_segmentation/releases/download/V1/create_lascar_json.json


Once the json file is downloaded, please place it in the same folder as the data.

### Data preparation
Processing the dataset by:
``` bash
python ./datasets/process_lascar.py
--data_dir <data-path>
--output_dir <output-path>
```


After processing is finished, the processed dataset will be save to output path


### Training


The following command can be used to run model training:
``` bash
python main.py
--json_list <json-path>
--data_dir <data-path>
--save_checkpoint 
--logdir=test 
--val_every 10 
--batch_size 2 
--max_epoch 200 
```

Note that you need to provide the location of your dataset directory by using ```--data_dir```.

To initiate distributed multi-gpu training, ```--distributed``` needs to be added to the training command.

To disable AMP, ```--noamp``` needs to be added to the training command.




### Testing
To evaluate the performance, the model path using `pretrained_dir` and model
name using `--pretrained_model_name` need to be provided:

```bash
python test.py 
--json_list=<json-path> 
--data_dir=<data-path>
--pretrained_model_name=<model-name> 
--pretrained_dir=<model-dir>
```
