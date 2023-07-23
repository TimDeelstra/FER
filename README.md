# Use of Facial Expression Recognition on video data to improve Social Robotics

This repository contains the code used for my Bachelor Thesis in which I apply Facial Expression Recognition on video data from the [SOROCOVA project](https://sorocova.nl/en/sorocova/).

[Link to the pretrained models](https://drive.google.com/drive/folders/1ODO24RpaiRb9QMjCk1gLADwfevUxFPC4?usp=sharing).

The models should be put into the repository in the same locations as in the directory structure of the Drive.


Recreating the environment can be done by using the environment.yml with Conda. The Python packages are also stored in requirements.txt to be used with other environments.
Since this Conda environment was setup for our specific system (RTX 4070, CUDA 11.7, Ubuntu 22.04) it might prove difficult to recreate the environment.
 ```
conda env create -f <environment-name>.yml
```

TODO:
- ~~Removed old Python scripts and models~~
- Remove legacy code from predict script
- Refactor predict script, easier to trace bugs and achieve intermediate results
- Update readme with command usage instructions
- Introduce a simple video player
- Allow for more control (introduce input parameters for hardcoded values)
- Introduce more options to analyse the results of each module in the pipeline.
- Reseperate scripts into modules, introducing a more modular approach.

- Rework facial image pre-processing, especially to prevent stretching the face too much.
- Introduce alternative [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) face detection, to drastically reduce the computational time of face detection (Retinaface detection reaches a max of 50-70 fps, while FER model can analyse 180-260 fps (not completely unexpected, since the facial images are less than 5% of the original image) while only slightly reducing performance.
