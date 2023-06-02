# Fitness-AQA (Fitness Action Quality Assessment) [ECCV'22]
Full paper: [Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment](https://arxiv.org/abs/2202.14019)

## Contents
1. [Introduction](https://github.com/ParitoshParmar/Fitness-AQA/edit/main/README.md#introduction)
2. [Our Self-Supervised Pose Contrastive Learning Approach for Fine-grained Action Assessment](https://github.com/ParitoshParmar/Fitness-AQA/edit/main/README.md#our-self-supervised-pose-contrastive-learning-approach-for-fine-grained-action-assessment)
3. [Our Self-Supervised Motion Disentangling Approach for Fine-grained Action Assessment](https://github.com/ParitoshParmar/Fitness-AQA#our-self-supervised-motion-disentangling-approach-for-fine-grained-action-assessment)
4. [Our Self-Supervised Pose and Appearance Disentangling Approach](https://github.com/ParitoshParmar/Fitness-AQA#our-self-supervised-pose-and-appearance-disentangling-approach)
5. [Our Method for Synchronizing In-the-Wild Videos](https://github.com/ParitoshParmar/Fitness-AQA/edit/main/README.md#our-method-for-synchronizing-in-the-wild-videos)
6. [Fitness-AQA Dataset for Fine-grained Exercise Action Quality Assessment](https://github.com/ParitoshParmar/Fitness-AQA#fitness-aqa-dataset-for-fine-grained-exercise-action-quality-assessment)

## Introduction
Analyzing a person's posture during exercising is necessary to prevent injuries and maximizing muscle mass gains. In this work, we present an AI-based approach to detect errors in workout form. Our approach is particularly applicable/useful in  real world gym scenarios, where off-the-shelf pose estimators fail to effectively capture person's pose due to challenging factors like camera recording angles, clothing styles, occlusions from gym equipment, etc. We applied our system to detect posture errors in three exercises: 1) BackSquat; 2) OverheadPress; and 3) BarbellRow. For this we collected the largest <b>fine-grained exercise action quality assessment</b> dataset, Fitness-AQA. Details on our self-supervised representation learning approaches and dataset are as follows:

## Our Self-Supervised Pose Contrastive Learning Approach for Fine-grained Action Assessment
<p align="left"> <img src="imgs/pose_contrastive_framework_2.png?raw=true" alt="cvcspc" width="400"/> </p>

## Our Self-Supervised Motion Disentangling Approach for Fine-grained Action Assessment
<p align="left"> <img src="imgs/approach_md.gif?raw=true" alt="motion_disentangling" width="400"/> </p>

## Our Self-Supervised Pose and Appearance Disentangling Approach
<p align="left"> <img src="imgs/swapping_approach_2_1.png?raw=true" alt="pose_appearance_disentangling" width="400"/> </p>

## Our Method for Synchronizing In-the-Wild Videos
<p align="left"> <img src="imgs/quasi_sync.PNG?raw=true" alt="video_quasi_syncing_technique" width="400"/> </p>

## Fitness-AQA Dataset for Fine-grained Exercise Action Quality Assessment
<p align="left"> <img src="imgs/exercise_dataset_hierarchy_3.png?raw=true" alt="fitness-aqa_dataset" width="800"/> </p>

Dataset available from: https://forms.gle/PbPTX1eVxGpa3QG88. Please send us the request to access the dataset using this form. By requesting the dataset, you agree to the [terms and conditions of usage](https://github.com/ParitoshParmar/Fitness-AQA/blob/main/fitness_aqa_dataset_license.pdf). This dataset shall only be used for non-commercial purposes. Please check your spam folder in case you seem to not have received the access after requesting it; or please contact me if you are still have not received the access. Thank you!

Please feel free to reach out to me if you have any questions or face any problems.

If you find our work useful, please consider citing:
```
@inproceedings{parmar2022domain,
  title={Domain Knowledge-Informed Self-supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXVIII},
  pages={105--123},
  year={2022},
  organization={Springer}
}
```
