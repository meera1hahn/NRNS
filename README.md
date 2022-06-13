# No RL No Simulation (NRNS)
Official implementation of the NRNS [paper](https://arxiv.org/abs/2110.09470): No RL, No Simulation: Learning to Navigate without Navigating

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
python download_aux.py
```

### Download Scene Data

Like Habitat-Lab, we expect a `data` folder (or symlink) with a particular structure in the top-level directory of this project. Running the `download_aux.py` script will download the pretrained models but you will still need to download the scene data. We evaluate our agents on Matterport3D (MP3D) and Gibson scene reconstructions. Instructions on how to download RealEstate10k can be found [here](https://google.github.io/realestate10k/download.html).

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


## Running pre-trained models
Look at the run scripts in src/image_nav/run_scripts/ for examples of how to run the model.

Difficulty settings options are: easy, medium, hard

Path Type setting options are: straight, curved

To run NRNS on gibson without noise for example on the straight setting with a medium difficulty

```
cd src/image_nav/
python -W ignore run.py \
    --dataset 'gibson' \
    --path_type 'straight' \
    --difficulty 'medium' \
```

### Scores
Expected scores for NRNS from running this code without noise

#### Gibson 

|Path Type |Success|SPL|
|------|--|--|
| Easy - Straight   | 0.7010 | 0.6271
| Medium - Straight | 0.5070 | 0.4153 
| Hard - Straight   | 0.2122 | 0.1547
| Easy - Curved     | 0.3140 | 0.1070
| Medium - Curved | 0.2200 | 0.0829
| Hard - Curved   | 0.1190 | 0.0544

#### Matterport

|Path Type|Success|SPL|
|------|--|--|
| Easy - Straight   | 0.5690 | 0.4926
| Medium - Straight | 0.3370 | 0.2717
| Hard - Straight   | 0.1960 | 0.1444
| Easy - Curved     | 0.2310 | 0.1086
| Medium - Curved | 0.1510 | 0.0738 
| Hard - Curved   | 0.0840 | 0.0414

## Citing

If you use NRNS in your research, please cite the following [paper](https://arxiv.org/abs/2110.09470):

```tex
@inproceedings{hahn_nrns_2021,
  title={No RL, No Simulation: Learning to Navigate without Navigating},
  author={Meera Hahn and Devendra Chaplot and Mustafa Mukadam and James M. Rehg and Shubham Tulsiani and Abhinav Gupta},
  booktitle={Neurips},
  year={2021}
 }
```
