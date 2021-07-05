import os
import cv2
import glob
import argparse
import numpy as np
from scipy.io import loadmat

def draw_3d(cropped_img, recon_img, image_name, save_dir):
    recon_mask: np.ndarray = recon_img[:,:,-1] * 255
    recon_mask = recon_mask.astype(np.uint8)
    recon_mask_inv: np.ndarray = 255 - recon_mask

    cropped_img = cropped_img[:, :, :3].astype(np.uint8)
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    recon_img = recon_img[:, :, :3].astype(np.uint8)
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

    img_bg = cv2.bitwise_and(cropped_img, cropped_img, mask=recon_mask_inv)
    img_fg = cv2.bitwise_and(recon_img, recon_img, mask=recon_mask)
    overlay = cv2.add(img_bg, img_fg)

    cv2.imwrite(os.path.join(save_dir, f"{image_name}.png"), overlay)
    return overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep3d_mat_dir", "-d", dest="deep3d_mat_dir", default=r"C:\Users\Mac\Documents\RawData\deep3d_mesh\output", help="read xxx.mat")
    parser.add_argument("--save_dir", "-s", dest="save_dir", default=r"C:\Users\Mac\Documents\RawData\deep3d_mesh\draw\3d", help="folder path to save the images")
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    for mat_path in glob.glob(os.path.join(opt.deep3d_mat_dir, "*.mat")):
        mat_dict = loadmat(mat_path)
        draw_3d(mat_dict["cropped_img"], mat_dict["recon_img"], os.path.splitext(os.path.basename(mat_path))[0], opt.save_dir)
