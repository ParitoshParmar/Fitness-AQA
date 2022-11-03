# Fitness-AQA
Full paper: [Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment](https://arxiv.org/abs/2202.14019)

## Introduction
Analyzing a person's posture during exercising is necessary to prevent injuries and maximizing muscle mass gains. In this work, we present a computer vision-based approach to detect errors in workout form. Our approach is particularly applicable/useful in  real world gym scenarios, where off-the-shelf pose estimators fail to effectively capture person's pose due to challenging factors like camera recording angles, clothing styles, occlusions from gym equipment, etc. We applied our system to detect posture errors in three exercises: 1) BackSquat; 2) OverheadPress; and 3) BarbellRow. For this we collected the largest exercise quality assessment dataset, Fitness-AQA. Details on our self-supervised representation learning approaches and dataset are as follows:

## Our Self-Supervised Pose Contrastive Learning Approach
<p align="left"> <img src="imgs/pose_contrastive_framework_2.png?raw=true" alt="cvcspc" width="400"/> </p>

## Our Self-Supervised Motion Disentangling Approach
<p align="left"> <img src="imgs/approach_md.gif?raw=true" alt="motion_disentangling" width="400"/> </p>

## Our Self-Supervised Pose and Appearance Disentangling Approach
<p align="left"> <img src="imgs/swapping_approach_2_1.png?raw=true" alt="pose_appearance_disentangling" width="400"/> </p>

## Fitness-AQA Dataset
<p align="left"> <img src="imgs/exercise_dataset_hierarchy_3.png?raw=true" alt="fitness-aqa_dataset" width="800"/> </p>

Dataset available from: https://forms.gle/PbPTX1eVxGpa3QG88. Please send us the request to access the dataset using this form.

If you find our work useful, please consider citing:
```
@article{parmar2022domain,
  title={Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  journal={arXiv preprint arXiv:2202.14019},
  year={2022}
}
```
