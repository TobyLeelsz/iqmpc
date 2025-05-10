# Reward-free World Models for Online Imitation Learning
Official Code Implementation of ICML 2025 paper: Reward-free World Models for Online Imitation Learning [Paper](https://arxiv.org/abs/2410.14081)

Shangzhe Li, Zhiao Huang, Hao Su

![demo_IQMPC](/images/combined_vertical_video.gif)

## (WIP) Expert Datasets are coming soon!

## Environment Setup and Running the Code

1. Setup the environment using the following commands:
```
conda env create -f conda_env/environment.yaml
conda activate iqmpc
```
2. Set the task in config.json and the correct expert dataset correspnding to the task.
3. Run the training code:
```
python3 tdmpc2/train.py
```
## Acknowledgement

This repository is created based on the original TD-MPC2 implementation repository: [TD-MPC2 Official Implementation](https://github.com/nicklashansen/tdmpc2).

## Citation

If you find our work helpful to your research, please consider citing our paper as follows:
```
@inproceedings{li2025reward,
  title={Reward-free World Models for Online Imitation Learning},
  author={Shangzhe Li and Zhiao Huang and Hao Su},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
