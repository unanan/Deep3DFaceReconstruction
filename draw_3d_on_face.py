import os
import cv2
import glob
import argparse
import numpy as np
import shutil
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

def draw_3d_by_two_images(cropped_img, recon_img:np.ndarray, image_name, save_dir):
    recon_gray = cv2.cvtColor(recon_img, cv2.COLOR_BGR2GRAY)
    recon_mask = np.zeros(recon_gray.shape, np.uint8)
    recon_mask[recon_gray.nonzero()] = 255

    recon_mask = recon_mask.astype(np.uint8)
    recon_mask_inv: np.ndarray = 255 - recon_mask

    img_bg = cv2.bitwise_and(cropped_img, cropped_img, mask=recon_mask_inv)
    img_fg = cv2.bitwise_and(recon_img, recon_img, mask=recon_mask)
    overlay = cv2.add(img_bg, img_fg)

    cv2.imwrite(os.path.join(save_dir, f"{image_name}.png"), overlay)
    return overlay


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep3d_mat_dir", "-d", dest="deep3d_mat_dir", default=r"C:\Users\Mac\Documents\RawData\wei_01\3d_checkpoints\20\deep3d", help="read xxx.mat")
    parser.add_argument("--cropped_img_dir", "-c", dest="cropped_img_dir", default=r"C:\Users\Mac\Documents\RawData\wei_01\crop", help="read xxx.mat")
    parser.add_argument("--recon_img_dir", "-", dest="recon_img_dir", default=r"C:\Users\Mac\Documents\RawData\wei_01\3d_checkpoints\20\render", help="read xxx.mat")
    parser.add_argument("--save_dir", "-s", dest="save_dir", default=r"C:\Users\Mac\Documents\RawData\wei_01\output\deep3d", help="folder path to save the images")
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    if opt.deep3d_mat_dir != None:
        for mat_path in glob.glob(os.path.join(opt.deep3d_mat_dir, "*.mat")):
            mat_dict = loadmat(mat_path)
            draw_3d(mat_dict["cropped_img"], mat_dict["recon_img"], os.path.splitext(os.path.basename(mat_path))[0], opt.save_dir)
        for cropped_img_path in glob.glob(os.path.join(opt.cropped_img_dir, "*.png")):
            cropped_img_name = os.path.basename(cropped_img_path)
            if not os.path.exists(os.path.join(opt.save_dir, cropped_img_name)):
                shutil.copy(cropped_img_path, os.path.join(opt.save_dir))
                print(cropped_img_path)

    else:
        for cropped_img_path in glob.glob(os.path.join(opt.cropped_img_dir, "*.png")):
            cropped_img_name = os.path.basename(cropped_img_path)
            recon_img_path = os.path.join(opt.recon_img_dir, cropped_img_name)

            draw_3d_by_two_images(cv2.imread(cropped_img_path), cv2.imread(recon_img_path), os.path.splitext(cropped_img_name)[0], opt.save_dir)
