# Video Interpolation: Rearranging frames in a video with smooth interpolation between each segment

This project is based on the below work:
```
@article{danier2023ldmvfi,
  title={LDMVFI: Video Frame Interpolation with Latent Diffusion Models},
  author={Danier, Duolikun and Zhang, Fan and Bull, David},
  journal={arXiv preprint arXiv:2303.09508},
  year={2023}
}
```

and is a fork of [LDMVFI](https://github.com/danier97/LDMVFI).

## Dependencies and Installation
See [environment.yaml](./environment.yaml) for requirements on packages. Simple installation:
```
conda env create -f environment.yaml
```

## Pre-trained Model
The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1_Xx2fBYQT9O-6O3zjzX76O9XduGnCh_7/view?usp=share_link), and its corresponding config file is [this yaml](./configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml).

\
To interpolate a video (in .yuv format), use the following code.
```
python interpolate_yuv.py \
--net LDMVFI \
--config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
--ckpt <path/to/ldmvfi-vqflow-f32-c256-concat_max.ckpt> \
--input_yuv <path/to/input/yuv> \
--size <spatial res of video, e.g. 1920x1080> \
--out_fps <output fps, should be 2 x original fps> \
--out_dir <desired/output/dir> \
--use_ddim
```

## Acknowledgement
My code is adapted from the original [LDMVFI](https://github.com/danier97/LDMVFI) repository. I thank the authors for sharing their code.