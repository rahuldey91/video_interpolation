import argparse
import torch
import torchvision.transforms.functional as TF
import os
from functools import partial
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.io import read_video, write_video


parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--net', type=str, default='LDMVFI')
parser.add_argument('--config', type=str, default='configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml')
parser.add_argument('--ckpt', type=str, default='ckpt.pth')
parser.add_argument('--input_file', type=str, default='input.mp4')
parser.add_argument('--interp_gap', type=int, default=5, help='num of frames before and after a new frame to be interpolated')

# sampler args
parser.add_argument('--use_ddim', dest='use_ddim', default=False, action='store_true')
parser.add_argument('--ddim_eta', type=float, default=1.0)
parser.add_argument('--ddim_steps', type=int, default=200)


def main():
    args = parser.parse_args()

    # initialise model
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = model.eval()
    print('Model loaded successfully')

    def interpolate(frame0, frame1):
        if len(frame0.shape) < 4:
            frame0 = frame0[None]
        if len(frame1.shape) < 4:
            frame1 = frame1[None]

        with torch.no_grad():
            with model.ema_scope():
                # form condition tensor and define shape of latent rep
                xc = {'prev_frame': frame0, 'next_frame': frame1}
                c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                shape = (model.channels, c.shape[2], c.shape[3])
                # run sampling and get denoised latent
                out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape)
                if isinstance(out, tuple): # using ddim
                    out = out[0]
                # reconstruct interpolated frame from latent
                out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]
        return out

    def binary_fill(frames, frame0_idx, frame1_idx):
        if frame0_idx >= frame1_idx-1:
            return
        new_idx = (frame0_idx + frame1_idx) // 2
        frame0 = frames[frame0_idx]
        frame1 = frames[frame1_idx]
        interp_frame = interpolate(frame0, frame1)[0]
        frames[new_idx] = interp_frame
        binary_fill(frames, frame0_idx, new_idx)
        binary_fill(frames, new_idx, frame1_idx)

    # set up sampler
    if args.use_ddim:
        ddim = DDIMSampler(model)
        sample_func = partial(ddim.sample, S=args.ddim_steps, eta=args.ddim_eta, verbose=False)
    else:
        sample_func = partial(model.sample_ddpm, return_intermediates=False, verbose=False)

    # Read input file
    stream, _, fps = read_video(args.input_file)
    fps = int(fps['video_fps'])

    # transform video to model readable form
    stream = (stream / 255.0) * 2 - 1
    stream = stream.permute(0, 3, 1, 2)
    stream = stream.cuda()

    # Setup output file
    _, fname = os.path.split(args.input_file)
    fname = fname.split('.')[0]

    # Start interpolation
    print('Using model {} to upsample file {}'.format(args.net, fname))

    # Original and new timestamps of frames to be rearranged
    frameA_orig_t = 16
    frameA_new_t = 5
    frameB_orig_t = 30
    frameB_new_t = 10
    frameA_orig_idx = frameA_orig_t * fps
    frameA_new_idx = frameA_new_t * fps
    frameB_orig_idx = frameB_orig_t * fps
    frameB_new_idx = frameB_new_t * fps
    last_idx = fps * 15
    interp_gap = args.interp_gap

    # initialize output frames as copy of input frames
    out_frames = stream[:last_idx].clone()

    # frame A inserted to new position
    out_frames[frameA_new_idx] = stream[frameA_orig_idx]

    # interpolate between first segment end and frameA
    binary_fill(out_frames, frameA_new_idx-interp_gap, frameA_new_idx)
    # interpolate between frameA and second segment start
    binary_fill(out_frames, frameA_new_idx, frameA_new_idx+interp_gap)

    # frame B inserted to new position
    out_frames[frameB_new_idx] = stream[frameB_orig_idx]

    # interpolate between second segment end and frame B
    binary_fill(out_frames, frameB_new_idx-interp_gap, frameB_new_idx)
    #interpolate between frameB and last segment start
    binary_fill(out_frames, frameB_new_idx, frameB_new_idx+interp_gap)

    # write output file
    out_frames = (out_frames * 0.5 + 0.5) * 255
    out_frames = out_frames.to(torch.uint8).permute(0,2,3,1).cpu()
    outname = '{}_interp_{}.mp4'.format(fname, interp_gap)
    write_video(outname, out_frames, fps=fps)
    print('Output written to {}'.format(outname))


if __name__ == "__main__":
    main()
