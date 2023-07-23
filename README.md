# Strivec: Sparse Tri-Vector Radiance Fields
<img src="image/USC-Logos.png" width=120px /><img src="./image/Adobe-Logos.png" width=120px /><img src="./image/ucsd_logo.png" width=60px />

[Project Sites](https://github.com/Zerg-Overmind/Strivec/)
 | [Paper](https://github.com/Zerg-Overmind/Strivec/)

## Overal Instruction
1. We build the initial geometry with the 1st stage of DVGO in our implementation by default, which is `use_geo = -1` in config files.
2. The geometry can be either initialized online (by default) or from other sources in `.txt` form, which can be enabled with `use_geo = 1` and `pointfile = /your/file.txt` in config files.  

For Synthetic-NeRF dataset, we provide the initial geometry from [DVGO](https://drive.google.com/file/d/1Z7grMvGNVUFa4RO1KuRAUkkL8JiHMMyw/view?usp=sharing) (which is the default one in our implementation) and from [MVS](https://drive.google.com/file/d/1m6ftmKU4lhxXQZKhkoeeWnC9F85kyMBu/view?usp=sharing). Feel free to try both (e.g., `use_geo = 1` and `pointfile = /your/mvs_file.txt`) to see the comparison. 

## Installation

### Requirements
All the codes are tested in the following environment:
* Linux  18.04+
* Python 3.6+
* PyTorch 1.10+
* CUDA 10.2+ 

## Data Preparation

* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Scannet](https://drive.google.com/drive/folders/1GoxJyf_YYEGvWStD7SpcPBqhePqCGpEJ)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Mip-NeRF360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)

And the layout should look like this:

```
Strivec
├── data
│   ├── nerf_synthetic
    │   │   │──chair
    │   │   │──drums
    │   │   │──...
    ├── scene0101_04 (scannet)
    │   │   │──exported
    │   │   │──scene0101_04_2d-instance-filt.zip
    │   │   │──...
    ├── scene0241_01 (scannet)
    │   │   │──exported
    │   │   │──scene0241_01_2d-instance-filt.zip
    │   │   │──...
    ├── TanksAndTemple
    │   │   │──Barn
    │   │   │──Caterpillar
    │   │   │──...
    ├── 360 (Mip-NeRF360)
    │   │   │──garden
    │   │   │──room
    │   │   │──...
```

## run training & evaluation
We not only provide the training and evaluation code to reproduce the results in the paper, but also the code of ablation that uses local VM tensors instead of local CP tensors (results
are [here](https://drive.google.com/drive/folders/1-OW0Qdnk4Wz-9BRr81P2mDe1aYDmjd0g?usp=sharing)).


```
# hierachical Strivec, without rotation (grid aligned)
python train_hier.py --config ./configs/synthetic-nerf/default/chair.txt

# local VM tensors instead of local CP tensors
train_dbasis.py --config ./configs/synthetic-nerf/local_vm/chair.txt

```

## Citation
If you find our code or paper helps, please consider citing:
```
@INPROCEEDINGS{gao2023iCCV,
  author = {Quankai Gao and Qiangeng Xu and Hao su and Ulrich Neumann and Zexiang Xu},
  title = {Strivec: Sparse Tri-Vector Radiance Fields},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```