<div align="center">
<h1>Activation Subspaces for Out-of-Distribution Detection</h1>

[**Baris Zöngür**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/visinf_team_details_141248.en.jsp)<sup>1</sup> &nbsp;&nbsp;&nbsp;
[**Robin Hesse**](https://robinhesse.github.io/)<sup>1</sup> &nbsp;&nbsp;&nbsp;
[**Stefan Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>1,2</sup>

<sup>1</sup>TU Darmstadt &nbsp;&nbsp;&nbsp;
<sup>2</sup>hessian.AI &nbsp;&nbsp;&nbsp;

<h3>ICCV 2025</h3>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](http://arxiv.org/abs/2508.21695)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## Abstract
To ensure the reliability of deep models in real-world applications, out-of-distribution (OOD) detection methods aim to distinguish samples close to the training distribution (in-distribution, ID) from those farther away (OOD). In this work, we propose a novel OOD detection method that utilizes singular value decomposition of the weight matrix of the classification head to decompose the model’s activations into decisive and insignificant components, which contribute maximally, respectively minimally, to the final classifier output. We find that the subspace of insignificant components more effectively distinguishes ID from OOD data than raw activations in regimes of large distribution shifts (Far-OOD). This occurs because the classification objective leaves the insignificant subspace largely unaffected, yielding features that are “untainted” by the target classification task. Conversely, in regimes of smaller distribution shifts (Near-OOD), we find that activation shaping methods profit from only considering the decisive subspace, as the insignificant component can cause interference in the activation space. By combining two findings into a single approach, termed ActSub, we achieve state-of-the-art results in various standard OOD benchmarks.

## Introduction

This repository contains two separate experimental settings under the following directories:

- **`actsub_openood`** – Contains the experimental setup for [OpenOOD](https://github.com/Jingkang50/OpenOOD) experiments reported in the main paper.  
- **`actsub_standard`** – Contains the experimental setup used for Tables 1, 2, and 3.

Before running the experiments, please complete the steps described in the **Setup** section below. Once finished, refer to each subdirectory for instructions specific to the corresponding experimental setting.

## Setup

### Environment

Create the conda environment and install the dependencies by running:
```bash
conda env create -f environment.yml
conda activate actsub
```

### Download
To download the training split of ImageNet-1k, run:
```bash
# Set the environment variable to the desired dataset directory. 
export DATASETS=<path_to_dataset_directory>
# Download and process the train split for ImageNet-1k.
bash download_train.sh
```

### Extract Training Samples

To extract the training samples for any specified backbone and experimental setting, run:
```bash
python extract_train_samples.py --config extract_config.yml
```
Please refer to the comments and parameters in **`extract_config.yml`** for configuration details, and modify them to match the desired backbone and experimental setting before running the script.

## References

Our code is based on [OpenOOD: Benchmarking Generalized OOD Detection](https://github.com/Jingkang50/OpenOOD) and [Extremely Simple Activation Shaping for Out-of-Distribution Detection](https://github.com/andrijazz/ash) repositories.

```bibtex
@inproceedings{sun2021react,
  title={ReAct: Out-of-distribution Detection With Rectified Activations},
  author={Sun, Yiyou and Guo, Chuan and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

@inproceedings{djurisic2023ash,
    author = {Djurisic, Andrija and Bozanic, Nebojsa and Ashok, Arjun and Liu, Rosanne},
    title = {Extremely Simple Activation Shaping for Out-of-Distribution Detection},
    booktitle = {Proceedings of the Eleventh International Conference on Learning Representations (ICLR 2023)},
    year = {2023},
}

@article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}
}

@inproceedings{xu2024scale,
  title={Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement},
  author={Xu, Kai and Chen, Rongyu and Franchi, Gianni and Yao, Angela},
  booktitle={Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)},
  year={2024}
}
```

      
## Citation

If you find our work useful, please consider citing our paper.

```bibtex
@inproceedings{zongur2025actsub,
  title = {Activation Subspaces for Out-of-Distribution Detection},
  author = {Baris Z{\"o}ng{\"u}r and Robin Hesse and Stefan Roth},
  booktitle = {Proceedings of the Twentieth IEEE International Conference on Computer Vision (ICCV 2025)},
  year = {2025},
}
```


