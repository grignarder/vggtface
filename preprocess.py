import os
import argparse
import numpy as np
import torch
import cv2

import facer
from pixel3dmm.lightning.p3dmm_system import system as p3dmm_system
from pixel3dmm import env_paths


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """img: HWC float32"""
    h, w = img.shape[:2]
    if h > w:
        y0 = (h - w) // 2
        return img[y0:y0 + w, :, :]
    else:
        x0 = (w - h) // 2
        return img[:, x0:x0 + h, :]


def imread_bgr_float01(path: str) -> np.ndarray:
    """cv2 BGR uint8 -> BGR float32 [0,1]"""
    bgr_u8 = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr_u8 is None:
        raise ValueError(f"Failed to read image: {path}")
    return bgr_u8.astype(np.float32) / 255.0


def float01_to_u8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='./examples/example1')
parser.add_argument('--output_folder', type=str, default='./preprocessed_data/example1')
args = parser.parse_args()

image_folder = args.image_folder
output_folder = args.output_folder
print(f'Processing images from {image_folder} and saving results to {output_folder}.')

mv_dir = os.path.join(output_folder, 'mv_image')
uv_dir = os.path.join(output_folder, 'uv')
mask_dir = os.path.join(output_folder, 'mask')
os.makedirs(mv_dir, exist_ok=True)
os.makedirs(uv_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Pixel3DMM
model = p3dmm_system.load_from_checkpoint(env_paths.CKPT_UV_PRED, strict=False).cuda().eval()

# facer
face_detector = facer.face_detector('retinaface/mobilenet', device='cuda')
face_parser = facer.face_parser('farl/celebm/448', device='cuda')
bad_indices = {0, 1, 3, 4, 5, 8, 9, 14, 16, 17, 18}

names = sorted([n for n in os.listdir(image_folder)
                if n.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not names:
    raise ValueError(f'No png/jpg images found in {image_folder}')

for name in names:
    stem = os.path.splitext(name)[0]
    in_path = os.path.join(image_folder, name)

    # ---- read -> float crop ----
    img_bgr = imread_bgr_float01(in_path)          # BGR float [0,1]
    img_bgr = center_crop_square(img_bgr)          # square crop on float

    # ---- mv_image: resize float -> save ----
    mv518_bgr = cv2.resize(img_bgr, (518, 518), interpolation=cv2.INTER_LINEAR)  # float
    mv_path = os.path.join(mv_dir, name)
    cv2.imwrite(mv_path, float01_to_u8(mv518_bgr))

    # ---- UV input: resize float -> RGB float ----
    img512_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)  # float
    img512_rgb = img512_bgr[:, :, ::-1].copy()  # RGB float [0,1]

    # (1,1,H,W,C)
    tar = torch.from_numpy(img512_rgb)[None, None].cuda()  # float32
    batch = {'tar_rgb': tar}

    # mirror ensemble
    batch_m = {'tar_rgb': torch.flip(tar, dims=[3])}  # flip W

    with torch.inference_mode():
        out, _ = model.net(batch)
        out_m, _ = model.net(batch_m)

        flipped = torch.flip(out_m['uv_map'], dims=[4])
        flipped[:, :, 0] *= -1
        flipped[:, :, 0] += 2 * 0.0075
        out['uv_map'] = (out['uv_map'] + flipped) / 2

    uv_map = torch.clamp((out['uv_map'][0, 0] + 1) / 2, 0, 1)
    uv_np = uv_map.permute(1, 2, 0).detach().cpu().numpy()
    np.save(os.path.join(uv_dir, f'{stem}_uv.npy'), uv_np)

    mv518_rgb = mv518_bgr[:, :, ::-1]  # float [0,1]
    mv_t = torch.from_numpy(mv518_rgb * 255.0).permute(2, 0, 1).unsqueeze(0).cuda()  # 1,3,518,518 float

    try:
        with torch.inference_mode():
            faces = face_detector(mv_t)
            faces = face_parser(mv_t, faces, bbox_scale_factor=1.25)

        seg_probs = faces['seg']['logits'].softmax(dim=1)  # nfaces x nclasses x h x w
        total = torch.zeros_like(seg_probs[0, 0])
        for i in range(seg_probs.size(1)):
            if i not in bad_indices:
                total += seg_probs[0, i]
        mask = total.detach().cpu().numpy()
    except Exception:
        mask = np.zeros((518, 518), dtype=np.float32)

    np.save(os.path.join(mask_dir, f'{stem}.npy'), mask)
