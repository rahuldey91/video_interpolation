# Video Interpolation: Rearranging frames in a video with smooth interpolation between each segment

This project is to address the following objective:

### Objective
Given a specific video footage (link), your task is to create a new 15-second interpolated video with the following specifications:

Frame A (captured at the 16-second mark of the original footage) should appear precisely at the 5-second mark in the new video.\
Frame B (captured at the 30-second mark of the original footage) should appear precisely at the 10-second mark in the new video.

### Key Requirements
Use interpolation techniques to connect segments of the footage, ensuring that Frame A and Frame B appear at their designated times (5-second and 10-second marks, respectively).
The transitions between different segments of the footage should be smooth and natural, without abrupt changes that could disrupt the flow of the video.
Maintain overall consistency in the video, including color, lighting, and motion, across the entire 15-second duration.

### Solution
My solution is based on the original LDMVFI (see acknowledgment below) work done by Danier <em>et al.</em> to interpolate between re-arranged frames in a video file. Since LDMVFI generates a single interpolated frame between two given frames, I use binary-search like algorithm to interpolate the gap between any two frames at locations `i` and `j`, i.e., it first generates the `(i+j)/2`-th frame, then it performs binary interpolation between `i` and `(i+j)/2` frames; and between `(i+j)/2` and `j` frames, and so on.

To find out the frames between which interpolation is to be done, I use optical flow between a reference frame (frameA/frameB) and frames in the other segments of the video. The optical flow is computed using [RAFT](https://github.com/princeton-vl/RAFT), and is currently computed pair-wise. An alternate warping based flow aggregation algorithm can be employed for faster flow computation.

A better result can be obtained by using a video-interpolation model that generates the whole chuck of frames betweeen `i` and `j` in a temporally consistent way. [VIDIM](https://vidim-interpolation.github.io/) looks like a promising approach, but they haven't released the source code yet.

## Dependencies and Installation
See [environment.yaml](./environment.yaml) for requirements on packages. Simple installation:
```
conda env create -f environment.yaml
```

## Inference
Download the pre-trained model from [here](https://drive.google.com/file/d/1_Xx2fBYQT9O-6O3zjzX76O9XduGnCh_7/view?usp=share_link) and put it into the `./pretrained` directory. Its corresponding config file is [this yaml](./configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml).

To run the re-arrange code:
```
python inference.py \
--net LDMVFI \
--config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
--ckpt pretrained/ldmvfi-vqflow-f32-c256-concat_max.ckpt \
--input_file input.mp4 \
--use_ddim \
--interp_gap 30
```

Here, `interp_gap` is the number of frames before and after an inserted frame (frameA) to do interpolation on. The higher the `interp_gap`, the smoother the interpolation and the longer the inference time.

### Results
Given the input `input.mp4`, the output videos are saved as `input_interp_<interp_gap>.mp4`. Some examples are:
\
* `input_interp_1.mp4` - this does not do any interpolation and you can see abrupt motion at the stiched frames.
* `input_interp_6.mp4` - less abrupt than before, but still high rate of motion around the stitched frames.
* `input_interp_20.mp4` - more smooth motion around the stitched frames, but there is some motion artifacts. The hands get blurred near the stitched frames.


### Speed
Since this solution used a per-frame diffusion interpolation model, it takes anywhere from <em>1-10</em> mins to finish depending on the interp_gap.

## Acknowledgement
This project is based on the below work:
```
Danier, Duolikun, Fan Zhang, and David Bull. "LDMVFI: Video Frame Interpolation with Latent Diffusion Models." AAAI, 2024.
```

Further, code is adapted from the original [LDMVFI](https://github.com/danier97/LDMVFI) repository. I thank the authors for sharing their code.