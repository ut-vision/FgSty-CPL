from __future__ import print_function
import argparse
import glob
import os.path as osp
import torch
import process_stylization
from photo_wct import PhotoWCT

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='pretrained_models/photo_wct.pth')
parser.add_argument('--content_image_path', default=None)
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default=None)
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--input_file', default="input_files/demo.txt")
parser.add_argument('--output_image_path', default='./results/example1.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--fg_only', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu.')
args = parser.parse_args()

img_exts = ['.jpg', '.jpeg', '.png']
# Load model
p_wct = PhotoWCT(gpu_id=args.gpu_id, fg_only=args.fg_only)
p_wct.load_state_dict(torch.load(args.model))

if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(args.gpu_id)

if args.content_image_path and args.style_image_path:
    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=args.content_image_path,
        style_image_path=args.style_image_path,
        content_seg_path=args.content_seg_path,
        style_seg_path=args.style_seg_path,
        output_image_path=args.output_image_path,
        cuda=args.cuda,
        gpu_id=args.gpu_id,
        save_intermediate=args.save_intermediate,
        no_post=args.no_post
    )
else: # load multiple files
    with open(args.input_file) as f:
        lines = f.readlines()
        for line in lines:
            args_list = line.split(",")
            content_image_path = args_list[0].strip()
            content_seg_path = args_list[1].strip()
            content_seg_path = None if content_seg_path == "none" else content_seg_path
            style_image_path = args_list[2].strip()
            style_seg_path = args_list[3].strip()
            style_seg_path = None if style_seg_path == "none" else style_seg_path
            output_image_path = args_list[4].strip()
            process_stylization.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_path,
                style_image_path=style_image_path,
                content_seg_path=content_seg_path,
                style_seg_path=style_seg_path,
                output_image_path=output_image_path,
                cuda=args.cuda,
                gpu_id=args.gpu_id,
                save_intermediate=args.save_intermediate,
                no_post=args.no_post
            )
