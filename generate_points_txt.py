import argparse
import os
import glob
from PIL import Image
import torch

from mlcandy.face_detection.retinaface_detector import RetinaFaceDetector

rf_detector = RetinaFaceDetector(
    "Resnet50_Final.pth",
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retina_weights_path", "-r", dest="retina_weights_path", default=r"Resnet50_Final.pth", help="")
    parser.add_argument("--image_dir", "-i", dest="image_dir", default=r"crop/", help="read xxx.mat")
    parser.add_argument("--save_dir", "-s", dest="save_dir", default=r"crop/", help="folder path to save the images")
    opt = parser.parse_args()

    for image_path in glob.glob(os.path.join(opt.image_dir, "*.png")):
        point_txt_path = os.path.join(
            opt.save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")

        image = Image.open(image_path)
        _, landmarks = rf_detector.detect(image)
        print(landmarks)
        with open(point_txt_path, "w") as f:
            for point in landmarks[0]:
                x, y = point
                f.write(f"{x}	{y}\n")
        f.close()
