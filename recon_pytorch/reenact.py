import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from models.resnet_50 import resnet50_use
from load_data import BFM, Preprocess
from reconstruction_mesh import reconstruction, transform_face_shape, estimate_intrinsic, render_img


class Deep3dReconDataset(Dataset):
    def __init__(self, image_dir, ref_lm3D):
        self.ref_lm3D = ref_lm3D
        self.image_paths = list(glob.glob(os.path.join(image_dir, "*.png")))

    @staticmethod
    def preprocess(img_path, ref_lm3D):
        pil_img = Image.open(img_path)

        # Get 5 landmarks
        #TODO
        lm_5 = None

        input_img_org, lm_new, transform_params = Preprocess(pil_img, lm_5, ref_lm3D)
        input_img = input_img_org.astype(np.float32)
        input_img_tensor = torch.from_numpy(input_img).permute(0, 3, 1, 2)

        # the input_img is BGR
        return input_img_tensor.squeeze(0), lm_5, transform_params

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_img_tensor, lm_5, transform_params = self.preprocess(self.image_paths[index], self.ref_lm3D)
        return input_img_tensor, lm_5, transform_params


class Deep3dReconstructor:
    def __init__(self, device):
        self.device = device
        self.isTrain = False #TODO
        self.model = resnet50_use().to(self.device)
        self.model.load_state_dict(torch.load(r'recon_pytorch/models/params.pt'))

        self.bfm = BFM(r'BFM/BFM_model_front.mat', self.device)
        self.ref_lm3D = self.bfm.load_lm3d()

        if self.isTrain:
            self.model.train()
        else:
            self.model.eval()

        # lm = [[122, 345], [], [], [], []]

    def reenact(self, coef, transform_params):
        face_shape, face_texture, face_color, landmarks_2d, z_buffer, angles, translation, gamma = reconstruction(coef, self.bfm)
        fx, px, fy, py = estimate_intrinsic(landmarks_2d, transform_params, z_buffer, face_shape, self.bfm, angles, translation)

        face_shape_t = transform_face_shape(face_shape, angles, translation)
        face_color = face_color / 255.0
        face_shape_t[:, :, 2] = 10.0 - face_shape_t[:, :, 2]

        image = render_img(face_shape_t, face_color, self.bfm, 300, fx, fy, px, py)
        image = image.detach().cpu().numpy()
        image = np.squeeze(image)

        image = np.uint8(image[:, :, :3] * 255.0)
        render_pil_img = Image.fromarray(image)
        return render_pil_img

    def forward(self, input_img_tensor):
        arr_coef_tensor = self.model(input_img_tensor)
        coef = torch.cat(arr_coef_tensor, 1)
        return coef


def arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="mode", dest="mode")

    single_parser = subparsers.add_parser("single", help='[MODE]single: test with single image')
    single_parser.add_argument("--image_path", "-i", dest="image_path", type=str, required=True)
    single_parser.add_argument("--save_path", "-s", dest="save_path", type=str, required=True)
    single_parser.add_argument("--num_workers", "-n", dest="num_workers", type=int, default=1)

    batch_parser = subparsers.add_parser("batch", help='[MODE]batch: test with batch of images')
    batch_parser.add_argument("--image_dir", "-i", dest="image_dir", type=str, required=True)
    batch_parser.add_argument("--save_dir", "-s", dest="save_dir", type=str, required=True)
    batch_parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, default=1)
    batch_parser.add_argument("--num_workers", "-n", dest="num_workers", type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    recon_model = Deep3dReconstructor(device)
    if args.mode == "single":
        input_img_tensor, lm_5, transform_params = Deep3dReconDataset.preprocess(args.image_path, recon_model.ref_lm3D)
        input_img_tensor = input_img_tensor.unsqueeze(0).to()
        coef_tensor = recon_model.forward(input_img_tensor)
        render_pil_image = recon_model.reenact(coef_tensor, transform_params)
        render_pil_image.save(args.save_path)
    else:
        # Build dataset
        recon_dataloader = DataLoader(
            Deep3dReconDataset(args.image_dir, recon_model.ref_lm3D),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=int(args.num_workers))

        count = 0
        for data in recon_dataloader:
            input_img_tensor, lm_5, transform_params = data
            coef_tensors = recon_model.forward(input_img_tensor)
            for coef_tensor in coef_tensors:
                render_pil_image = recon_model.reenact(coef_tensor, transform_params)
                render_pil_image.save(os.path.join(args.save_dir, f"{count}.png"))
                count += 1
