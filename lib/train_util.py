import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm

def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
        save_obj_mesh(save_path, verts, faces)

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')




