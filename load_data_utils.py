import torch
import numpy as np
import os
import cv2
from vggt.models.vggt import VGGT
import facer



# face_detector = facer.face_detector('retinaface/mobilenet', device="cuda")
# face_parser = facer.face_parser('farl/celebm/448', device="cuda")  # optional "farl/lapa/448"
# if we already have masks, we can skip the face_parser step
face_detector = None
face_parser = None

bad_indices = [
    0,  # background,
    1,  # neck
    # 2, skin
    3,  # cloth
    4,  # ear_r (images-space r)
    5,  # ear_l
    # 6 brow_r
    # 7 brow_l
    8,  # eye_r
    9,  # eye_l
    # 10 noise
    # 11 mouth
    # 12 lower_lip
    # 13 upper_lip
    14,  # hair,
    # 15, glasses
    16,  # ??
    17,  # earring_r
    18,  # ?
]


def load_data(base_path, keys, device="auto", erode_mask=False, return_tensor=False):
    global face_detector, face_parser
    for key in keys:
        if key not in ["imgs", "uvs", "masks"]:
            raise ValueError(f"Unsupported key: {key}. Supported keys are 'imgs', 'uvs', 'masks'.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_PATH = base_path
    image_names = os.listdir(os.path.join(BASE_PATH, "mv_image"))
    image_names.sort()
    # remove extension
    image_names = [os.path.splitext(name)[0] for name in image_names]
    print(image_names)
    data = {}
    for key in keys:
        data[key] = []
    if erode_mask:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
    if "imgs" in keys:
        for image_name in image_names:
            img = cv2.imread(os.path.join(BASE_PATH, "mv_image", f"{image_name}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/ 255.0  # Normalize to [0, 1]
            data["imgs"].append(img)
        data['imgs'] = np.stack(data['imgs'], axis=0)  # (N, H, W, C)
    if "uvs" in keys:
        for image_name in image_names:
            uv = np.load(os.path.join(BASE_PATH, 'uv', f"{image_name}_uv.npy")) # 512, 512, 2
            # resize uv to 518* 518 using bilinear interpolation
            uv = cv2.resize(uv, (518, 518), interpolation=cv2.INTER_LINEAR)
            # import random
            # uv = uv + (random.random() - 0.5)*0.03
            data["uvs"].append(uv)
        data['uvs'] = np.stack(data['uvs'], axis=0)  # (N, H, W, 2)
    if "masks" in keys:
        if not os.path.exists(os.path.join(BASE_PATH, 'mask')):
            os.makedirs(os.path.join(BASE_PATH, 'mask'))
            # If mask folder does not exist, use facer to predict masks
            if face_detector is None or face_parser is None:
                face_detector = facer.face_detector('retinaface/mobilenet', device="cuda")
                face_parser = facer.face_parser('farl/celebm/448', device="cuda")  # optional "farl/lapa/448"
            masks = get_mask_from_imgs(data['imgs'].transpose(0, 3, 1, 2)) # N, H, W
            print(masks.shape)
            # save masks
            for i, image_name in enumerate(image_names):
                mask = masks[i]
                np.save(os.path.join(BASE_PATH, 'mask', f"{image_name}.npy"), mask)
        for image_name in image_names:
            mask = np.load(os.path.join(BASE_PATH, 'mask', f"{image_name}.npy"))
            # normalize mask to [0, 1]
            mask = mask / mask.max()
            if erode_mask:
                mask = cv2.erode(mask, kernel, iterations=10)
            data["masks"].append(mask)
        data['masks'] = np.stack(data['masks'], axis=0)  # (N, H, W)
    if return_tensor:
        # Convert to torch tensors
        data["imgs"] = torch.from_numpy(data["imgs"]).to(device).float()
        data["uvs"] = torch.from_numpy(data["uvs"]).to(device).float()
        data["masks"] = torch.from_numpy(data["masks"]).to(device).float()
    return data


def get_mask_from_imgs(imgs):
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float().cuda()
        return_tensor = False
    else:
        return_tensor = True
    # normalize imgs to 0-255
    if imgs.max() > 10:
        pass
    else:
        imgs = imgs * 255.0
    try:
        with torch.inference_mode():
            faces = face_detector(imgs)
            torch.cuda.empty_cache()
            faces = face_parser(imgs, faces, bbox_scale_factor=1.25)
            torch.cuda.empty_cache()
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        total_masks = torch.zeros_like(seg_probs[:,0,:,:])
        for i in range(18):
            if i not in bad_indices:
                total_masks += seg_probs[:,i,...]
        # total_masks = total_masks.cpu().numpy()
    except:
        total_masks = torch.zeros((imgs.shape[0],518,518),device="cuda")
    if return_tensor:
        return total_masks
    else:
        return total_masks.detach().cpu().numpy()


def load_vggt_models(model_path="./pretrained_weights/vggt_weights.pt", device="auto"):
    """
    Load the VGGT model from the specified path.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    _URL = model_path
    model.load_state_dict(torch.load(_URL))
    model.eval()
    model = model.to(device)
    print("VGGT model loaded successfully.")
    return model