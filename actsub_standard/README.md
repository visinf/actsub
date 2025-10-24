## Download

Run the following commands in the **`actsub/actsub_standard`** directory to download the required datasets:

```bash
# Set the environment variable to the desired dataset directory. 
export DATASETS=<path_to_dataset_directory>
# Download and process the datasets required for evaluation.
bash download.sh
```

Some datasets hosted by [OpenOOD](https://github.com/Jingkang50/OpenOOD) may fail to download due to gdown access quota limits. If this occurs, run the following commands to recover any missing datasets:

```bash
export DATASETS=<path_to_dataset_directory>
bash download_recover.sh
```

## Evaluation

To evaluate ActSub, run the following command: 
```bash
python evaluate.py --config configs/eval_config.yml
```
For configuration details, refer to the comments and parameters in **`configs/eval_config.yml`**. The evaluation script reports OOD detection accuracy for all datasets under the specified in-distribution (ID) dataset and backbone, which can be selected by modifying the config file.

## Tuning

The tuning script reports the optimal parameters within a predefined range for the ActSub method and can be used as follows:
```bash
python tune.py --config configs/tune_config.yml
```

Configuration details are provided in **`configs/tune_config.yml`**, where parameters can be modified to adjust the tuning setup.

