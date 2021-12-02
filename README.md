# DFTS2
Deep Feature Transmission Simulator (DFTS) version 2.

Originally released in 2018 [[1]](#references), DFTS was developed to be compatible with Tensorflow version 1 (more specifically, version 1.12) and Keras 2.2.2. The demo paper [[2]](#references) gave a brief overview of the simulator. Various changes in Tensorflow 2 [[3]](#references) break the operation of DFTS. 

We have modified DFTS to be now fully Tensorflow version 2-compatible in this repository. Previously we edited the original DFTS to run (with minimal modification) in Tensorflow 2 by disabling the v2 behavior in [[4]](#references). 

DFTS2 is a sophisticated simulation framework. It has new features:
1. TensorFlow version 2 compatibility.
2. Additional communication channel models and simulation modes.
3. Missing feature recovery methods from the recent literature.

## Contents
- [Overview](#overview)
- [Publications](#publications)
- [Presentation and demonstration video on YouTube](#presentation-and-demonstration)
- [User documentation](#user-documentation)

## Overview
The following figure gives a system overview of Collaborative Intelligence strategies implemented in DFTS2.

<img src="https://github.com/AshivDhondea/dfts2_user_doc/blob/main/Figures/sytemoverviewclipped.png" width="400" height="400">

## Publications
Two peer reviewed conference papers were published on work done with DFTS2.
* A. Dhondea, R. A. Cohen, and I. V.Bajić, [**CALTeC: Content-adaptive linear tensor completion for collaborative intelligence**](https://ieeexplore.ieee.org/document/9506372), Proc. IEEE ICIP, 2021.
* A. Dhondea, R. A. Cohen, and I.V.Bajić, **DFTS2: Deep feature transmission simulator for collaborative intelligence**.
For benchmarking purposes and to assist future users, we provide our packet traces, example simulation scripts and Monte Carlo experiment result files in a [Dropbox directory](https://www.dropbox.com/home/MEng_Project/dfts2). The full-scale test set used in our experiments is the same subset of the Imagenet validation set from the original DFTS demo paper [[1]](#references).

## Presentation and demonstration
In this [YouTube video](https://www.youtube.com/watch?v=5dW5i2XeCd0), we present our simulator, demonstrate how to set up a Python virtual environment for DFTS2, and demonstrate how to use DFTS2.

## User documentation
The latest version of the user documentation manual can be found [[here](https://github.com/AshivDhondea/dfts2_user_doc)]

## References
### [1] Unnibhavi, H. (2018) DFTS (Version 1.0) [[repo](https://github.com/SFU-Multimedia-Lab/DFTS)]

### [2] H. Unnibhavi, H. Choi, S. R. Alvar, and I. V. Bajić, "DFTS: Deep Feature Transmission Simulator," demo paper at IEEE MMSP'18, Vancouver, BC, Aug. 2018. [[pdf](https://www.researchgate.net/publication/327477545_DFTS_Deep_Feature_Transmission_Simulator)]  

### [3] Effective TensorFlow 2 [[guide](https://www.tensorflow.org/guide/effective_tf2)]

### [4] Dhondea, A. (2020) DFTS_compat_v1 (Version 1.0) [[repo](https://github.com/AshivDhondea/DFTS_compat_v1)]

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/AshivDhondea/DFTS2/blob/master/LICENSE) file for details.
