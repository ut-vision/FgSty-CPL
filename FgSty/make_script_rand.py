import os, sys, glob
import numpy as np
import subprocess
import random

path2fname = lambda path: path.split("/")[-1].split(".")[0]
ver = "v1"
np.random.seed(seed=12345)
n_split = 5 # # of scripts
modes = ["test", "train"]
n_style_images_per_target = 10 
root = "/path/to/your/root"
#NOTE: choose which data you use as the style set
style_sets = ["GTEA"]
# style_sets = ["GTEA", "EDSH12", "UTG", "YHG"]
"""
content and style directories should be
- root / source-dataset (e.g., EGTEA, ObMan-Ego)
    - train
    - trainannot (segmentation mask)
    - test
    - testannot (segmentation mask)
- root / target-datasets (e.g., GTEA, EDSH12, UTG, YHG)
    - train
    - trainannot (segmentation mask)
"""
content_dirname = "EGTEA"
content_dir = os.path.join(root, content_dirname)
style_path = {"GTEA": {"img": [], "mask": []}, "EDSH12": {"img": [], "mask": []}, "UTG": {"img": [], "mask": []}, "YHG": {"img": [], "mask": []}}
sadb_suffix = "rand"
#NOTE: for ObMan-Ego, use "pretrained_models/photo_wct_sim.pth"
pretrained_model = "pretrained_models/photo_wct.pth"

os.makedirs("scripts", exist_ok=True)
os.makedirs("input_files/%s"%ver, exist_ok=True)

for style_set in style_sets:
    img_files = sorted(glob.glob(os.path.join(root,"%s/train/*.png"%style_set)) + glob.glob(os.path.join(root,"%s/train/*.jpg"%style_set)))
    seg_files = sorted(glob.glob(os.path.join(root, "%s/trainannot/*.png" % style_set)) + glob.glob(os.path.join(root, "%s/trainannot/*.jpg" % style_set)))
    assert len(img_files) == len(seg_files)
    print(style_set, len(img_files))
    all_idx = list(range(0, len(img_files)))
    if n_style_images_per_target < len(img_files):
        choice_idx = np.random.choice(all_idx, n_style_images_per_target, replace=False)  # choose n_style images
    else:
        choice_idx = all_idx 
    print(len(choice_idx))
    for idx in choice_idx:
        style_path[style_set]["img"].append(img_files[idx])
        style_path[style_set]["mask"].append(seg_files[idx])

for mode in modes:
    content_img_files = sorted(glob.glob("%s/%s/*.png"%(content_dir, mode)) + glob.glob("%s/%s/*.jpg"%(content_dir, mode)))
    content_seg_files = sorted(glob.glob("%s/%s/*.png"%(content_dir, mode+"annot")) + glob.glob("%s/%s/*.jpg"%(content_dir, mode+"annot")))
    print("source size", len(content_img_files))
    n_split_unit = np.ceil(len(content_img_files)/n_split)
    assert len(content_img_files) == len(content_seg_files)
    os.makedirs(os.path.join("results", content_dirname + "_%s_%s" % (sadb_suffix, ver), mode), exist_ok=True)
    os.makedirs(os.path.join("results", content_dirname + "_%s_%s" % (sadb_suffix, ver), mode + "annot"), exist_ok=True)
    for j, (content_image_path, content_seg_path) in enumerate(zip(content_img_files, content_seg_files)):
        set_id = np.random.randint(0, len(style_sets), 1)[0]
        style_set = style_sets[set_id] # choose style dir
        k = int(j // float(n_split_unit)) + 1
        file_id = np.random.randint(0, len(style_path[style_set]["img"]), 1)[0]
        style_image_path = style_path[style_set]["img"][file_id] # pick style images randomly
        style_seg_path = style_path[style_set]["mask"][file_id] 
        out_filename = "%s_stylized_by_%s_%s.png"%(path2fname(content_image_path), style_set, path2fname(style_image_path))
        output_image_path = os.path.join("results", content_dirname + "_%s_%s" % (sadb_suffix, ver), mode, out_filename)
        with open("input_files/%s/%s_%s_%s_part%03d.txt"%(ver, content_dirname, ver, mode, k), "a") as f:
            f.write(",".join([content_image_path, content_seg_path, style_image_path, style_seg_path, output_image_path, "\n"]))
        subprocess.call(["cp", content_seg_path, os.path.join("results", content_dirname + "_%s_%s" % (sadb_suffix, ver), mode + "annot", out_filename)])
    for l in range(1, n_split+1):
        with open("scripts/%s_%s_%s_part%03d.sh" % (content_dirname, ver, mode, l), "w") as f:
            f.write(" ".join(["python", "test.py", "--model", pretrained_model, "--input_file", "input_files/%s/%s_%s_%s_part%03d.txt" % (ver, content_dirname, ver, mode, l), "\n"]))
    print("done", mode)
