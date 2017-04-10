

# CRAFT

## Overview

This is the matlab implementation for [”Person Re-Identification by Camera Correlation Aware Feature Augmentation"][http://ieeexplore.ieee.org/document/7849147/].

## Files

| Files                | Description                              |
| -------------------- | ---------------------------------------- |
| demo.m               | The entrance script. Run it for demonstration. |
| CRAFT.m              | Transform feasture (either in the original space or in the kernel space) to the augmented ones. |
| cameraCorrelation.m  | Estimate the camera correlation. It is used in CRAFT.m. |
| MFA_class.m          | The MFA distance matrix learning.        |
| evaluation.m         | A function that evaluates the performance. |
| getCMC.m             | Compute CMC.                             |
| train_test_split.mat | The training and testing split used in our paper. |
| viper.mat            | The ViPeR dataset.                       |

## How to use

"demo.m" is the entrance script. Please run it with matlab to see the pipeline of our system. You can use "matlabpool" for parallel computing.

"cameraCorrelation.m" estimates the camera correlation. "CRAFT.m" transforms the original feature by CRAFT. These two functions are the main contribution of our system. 

## Citation

Ying-Cong Chen, Xiatian Zhu,Wei-Shi Zheng, and Jian-Huang Lai. Person Re-Identification by Camera Correlation Aware Feature Augmentation. IEEETransactions on Pattern Analysis and Machine Intelligence (DOI: 10.1109/TPAMI.2017.2666805)