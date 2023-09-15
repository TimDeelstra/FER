# Use of Facial Expression Recognition on video data to improve Social Robotics

This repository contains the code used for my Bachelor Thesis in which I apply Facial Expression Recognition on video data from the [SOROCOVA project](https://sorocova.nl/en/sorocova/).

[Link to the pretrained models](https://drive.google.com/drive/folders/1ODO24RpaiRb9QMjCk1gLADwfevUxFPC4?usp=sharing).

The models should be put into the repository in the same locations as in the directory structure of the Drive.


Recreating the environment can be done by using the environment.yml with Conda. The Python packages are also stored in requirements.txt to be used with other environments.
Since this Conda environment was setup for our specific system (RTX 4070, CUDA 11.7, Ubuntu 22.04) it might prove difficult to recreate the environment.
 ```
conda env create -f <environment-name>.yml
```

## Use of Framework

predict.py

Run the FER model on a single video
```
python3 predict.py
-d <maindir> 'the directory to store the data files'
-f <videofile> 'the video to analyse'
-m <model_number> 'the model to be used'
-b <backend_number> 'the backend to be used'
-s <batch_size> 'the amount of image to process as a batch' Default(8)
-r(render) 'whether to display the FER predicitions on the video in a window' Optional
-p(realtime playback) 'playback the FER predictions in realtime' Optional
-v(verbose) 'output some additional information' Optional
```

batch_predict.py

Run the prediction for model 3,4,5 on the sessions from the csv file that indicates errors
```
python3 batch_predict.py <error_file> <video_dir>
<error_file> 'file containing the list of sessions and their errors'
<video_dir> 'the directory containing the videos from these sessions'
```

detect.py

Use the detection method on the predict data to convert to an event-level
```
python3 detect.py <predict_file> <video_file>
<predict_file> 'the predict file'
<video_file> 'the video corresponding to the predict file'
```

batch_detect.py

Use the detection method on the predict data of a batch of videos to convert to an event-level
```
python3 batch_detect.py <predict_dir> <video_dir> <render>
<predict_dir> 'directory containing the predict files'
<video_dir> 'directory containing the videos of the predictions'
<render> 'whether to render a compilation of the detections for each video'
```

playback.py

Display the emotion data on a video
```
python3 playback.py <video_file> <data_file>
<video_file> 'the video file'
<data_file> 'the file containing the emotion data
```

## Models
| Index | Model |
|---|---|
| 1 | [HSE](https://github.com/HSE-asavchenko/face-emotion-recognition) |
| 2 | [RMN](https://pypi.org/project/rmn/) |
| 3 | [POSTER V2 AffectNet](https://github.com/Talented-Q/POSTER_V2) |
| 4 | [POSTER V2 RAF-DB](https://github.com/Talented-Q/POSTER_V2) |
| 5 | [APViT](https://github.com/youqingxiaozhua/APViT) |

TODO:
- ~~Remove old Python scripts and models~~
- Remove legacy code from predict script
- Refactor predict script, easier to trace bugs and achieve intermediate results
- ~~Update readme with command usage instructions~~
- ~~Introduce a simple video player~~
- Allow for more control (introduce input parameters for hardcoded values)
- Introduce more options to analyse the results of each module in the pipeline.

- Rework facial image pre-processing, especially to prevent stretching the face too much.
- Introduce alternative [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) face detection, to drastically reduce the computational time of face detection (Retinaface detection reaches a max of 50-70 fps, while FER model can analyse 180-260 fps (not completely unexpected, since the facial images are less than 5% of the original image) while only slightly reducing performance.
