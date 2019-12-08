Noise Flow - A normalizing flows model for image noise modeling and synthesis
===

This repository provides the codes for training and testing the Noise Flow model used for image noise modeling and 
synthesis as described in the paper:

[**Noise Flow: Noise Modeling with Conditional Normalizing Flows**](https://arxiv.org/pdf/1908.08453.pdf)

It also provides code for training and testing a CNN-based image denoiser (DnCNN) using Noise Flow as a noise generator, with comparison to other noise generation methods (i.e., AWGN and signal-dependent noise).
  
# Required libraries

Python (works with 3.6)

TensorFlow (works with 2.0)

TensorFlow Probability (tested with 0.8.0)

_Despite not tested, the code may work with library versions other than the specified._

# Required dataset

[Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/)

It is recommended to use the medium-size SIDD for training Noise Flow:

[SIDD_Medium_Raw](http://bit.ly/2kHT7Yr)

The code checks for and downloads `SIDD_Medium_Raw` if it does not exist. 

# Training/Testing/Sampling

Start by running `train.ipynb`

It contains a set of examples for training different models (as described in the paper) and optionally perform testing and 
sampling concurrently.

## goal for this repository

the origianl codes provided by the author have some running problem, and the pretrained model seems to not work well. 
Therefore, I do some modification based on the original codes and rewrite the training codes which estimate the algorithm 
performance using KL between the real noise image and the generating noise image. After a serise of experiment, I find this 
algorithm is limit for specific cam and iso.

   
