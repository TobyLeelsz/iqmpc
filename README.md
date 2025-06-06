# Reward-free World Models for Online Imitation Learning
Official code implementation of ICML 2025 paper: **Reward-free World Models for Online Imitation Learning**   [[Paper Link]](https://arxiv.org/abs/2410.14081)

Shangzhe Li, Zhiao Huang, Hao Su

![demo_IQMPC](/images/combined_vertical_video.gif)

##  Introduction

IQ-MPC is a world model designed for online imitation learning. It leverages the inverse soft-Q objective to train the critic, enabling effective policy learning from limited expert demonstrations and online reward-free interactions. Built upon the architecture of [TD-MPC2](https://www.tdmpc2.com/), IQ-MPC excels in handling complex tasks such as dexterous hand manipulation and high-dimensional locomotion.

## Environment Setup and Running the Code

1. Setup the environment using the following commands:
```
conda env create -f conda_env/environment.yaml
conda activate iqmpc
```
2. Download the expert datasets [here](https://drive.google.com/drive/folders/1d_0ks7Ion9onWrWEX9JBGDiaB7oNB6da?usp=sharing), which includes the expert datasets for 6 locomotion tasks and 3 dexterous hand manipulation tasks. All of the expert demonstrations are sampled from a trained single-task TD-MPC2 agent.
3. Set the task in tdmpc2/config.json and the correct expert dataset path corresponding to the task.
4. Run the training code:
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
