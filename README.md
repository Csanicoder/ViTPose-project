# :climbing: Climbing Movement Analysis
This proof of concept project for implementing an AI assisted tool that can help climbers analyse their movement.

## Goals
My main idea is to use the keypoints of the body detected by the pose recognition agent, along with either:
- Another **agent** that is trained to analyze the efficiency of the movement produced by ViTPose.\
*(I think the gradient descent algorithm could be used to train a model like this, the cost function being the difference between an optimal climb and a beginner / less optimal one.)*
- Or some kind of **algorithm** *(with identical purpose to the trained agent)*.\
I had some loose ideas for this, for example using the center of mass of the body, and potentially figuring out the forces acting on it.

Ultimately, the result should contain **advice** on which parts of the movement are good as well as which ones can be improved upon and how.

## Current State

Currently, I implemented the usage of a ViTPose model to detect keypoints on a **video** input. The output is the same video, with the keypoints drawn on top of the person.

As I update the project, I'll keep updating this section as well.

## Dependecies
The necessary python libraries are listed in *'requirements.txt'*.\
The following models are used from hugging-face:
- [PekingU/rtdetr_r50vd_coco_o365](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) - for detecting people
- [usyd-community/vitpose-base-simple](https://huggingface.co/usyd-community/vitpose-base-simple) - for detecting keypoints on these people

## Project structure

Currently, there is one directory (*'src/'*) containing all program files. The *'pose/'* directory contains modules for processing and drawing on the image / frame.

## How to run

If you want to run the model on a single image, run *'vitpose_image_demo.py'*.

For the video version, run *'vitpose_video_demo.py'*.
The input path of both versions is given in the source code.

**You want to change the value of the path variables ```image_path``` and ```video_path``` to an existing file on your computer!**

You can download climbing videos from [this](https://drive.google.com/drive/folders/1g5j74eu0UsE0nyM0RjFE1hUrgHc-h7IP?usp=sharing) drive directory.

## Output

The ouput is an image / video file, identical to the input, except that the keypoints are drawn on top of them.\
In the image demo, the path is hard coded into the ```cv2.imwrite()``` function to ```../out/image/annotated_output.jpg```.\
In the video demo, the path is defined in the ```output_path``` variable as ```../out/video/annotated_video_output.mp4```.