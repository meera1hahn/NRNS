# No RL No Simulation (NRNS)
Official implementation of the NRNS paper: No RL, No Simulation: Learning to Navigate without Navigating

NRNS is a heriarchical modular approach to image goal navigation that uses a topological map and distance estimator to navigate and self-localize. Distance function and target prediction function are learnt over passive video trajectories gathered from Mp3D and Gibson.

NRNS is a heriarchical modular approach to image goal navigation that uses a topological map and distance estimator to navigate and self-localize. Distance function and target prediction function are learnt over passive video trajectories gathered from Mp3D and Gibson.

[[project website](https://meerahahn.github.io/nrns)]

## Setup

This project is developed with Python 3.6. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:

```bash
conda create -n nrns python3.6
conda activate nrns
```

### Install Habitat and Other Dependencies

NRNS makes extensive use of the Habitat Simulator and Habitat-Lab developed by FAIR. You will first need to install both [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). 

Please find the instructions to install habitat [here](https://github.com/facebookresearch/habitat-lab#installation)

If you are using conda, Habitat-Sim can easily be installed with

```bash
conda install -c aihabitat -c conda-forge habitat-sim headless
```

We recommend downloading the test scenes and running the example script as described [here](https://github.com/facebookresearch/habitat-lab/blob/v0.1.5/README.md#installation) to ensure the installation of Habitat-Sim and Habitat-Lab was successful. Now you can clone this repository and install the rest of the dependencies:

```bash
git clone git@github.com:meera1hahn/NRNS.git
cd NRNS
python -m pip install -r requirements.txt
```

### Download Data

Like Habitat-Lab, we expect a `data` folder (or symlink) with a particular structure in the top-level directory of this project. We evaluate our agents on Matterport3D (MP3D) and Gibson scene reconstructions.



#### Image-Nav Test Episodes
The image-nav test epsiodes used in this paper for MP3D and Gibson can be found [here.](https://meerahahn.github.io/nrns/data) These were used to test all baselines and NRNS.


#### Matterport3D

The official Matterport3D download script (`download_mp.py`) can be accessed by following the "Dataset Download" instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded this way:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract this data to `data/scene_datasets/mp3d` such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 total scenes. We follow the standard train/val/test splits. 

#### Gibson 

The official Gibson dataset can be accessed on their [project webpage](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md). Please follow the link to download the Habitat Simulator compatible data. The link will first take you to the license agreement and then to the data. We follow the standard train/val/test splits. 


## Citing

If you use NRNS in your research, please cite the following [paper](https://arxiv.org/abs/2004.02857):

```tex
@inproceedings{hahn_nrns_2021,
  title={No RL, No Simulation: Learning to Navigate without Navigating},
  author={Meera Hahn and Devendra Chaplot and Mustafa Mukadam and James M. Rehg and Shubham Tulsiani and Abhinav Gupta},
  booktitle={Arxiv},
  year={2021}
 }
```
