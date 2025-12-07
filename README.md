# Implicit Neural Representations for Respiratory Motion Estimation from X-ray Coronary Angiography

## About this work
We propose a conditional implicit neural representation (cINR) method to recover and disentangle the primary sources of variation in coronary angiogram sequences: cardiac motion, respiratory motion, and contrast intensity changes.
This is with the ultimate goal of recovering respiratory motion from coronary angiograms, useful for clinical applications like dynamic coronary roadmapping.
Our method represents these sequences as INRs conditioned on three learned latent codes to capture the underlying motion signals of individual frames.
Disentanglement of the signals is enforced through regularization, including Lipschitz regularization, periodicity regularization, and smoothness regularization.
Our method is fully self-supervised, and operates directly on individual XCA sequences.

## Repository
This repository contains the implementation of the PyTorch models and re-implementation of the state-of-the-art manifold methods. We utilize three datasets in our work, of which two are private. The public dataset, XCAV, can be found through the authors' website [here](https://kirito878.github.io/DeNVeR/). We utilize Weights & Biases for logging purposes.