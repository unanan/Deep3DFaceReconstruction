import argparse
import os
import glob
from PIL import Image

from mlcandy.face_detection.Pytorch_Retinaface.detect import retinaface_detect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", "-i", dest="image_dir", default=r"crop/", help="read xxx.mat")
    parser.add_argument("--save_dir", "-s", dest="save_dir", default=r"crop/", help="folder path to save the images")
    opt = parser.parse_args()

    for image_path in glob.glob(os.path.join(opt.image_dir, "*.png")):
        point_txt_path = os.path.join(
            opt.save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")

        image = Image.open(image_path)
        _, landmarks = retinaface_detect(image)

        with open(point_txt_path, "w") as f:
            for point in landmarks[0]:
                x, y = point
                f.write(f"{x}	{y}\n")
        f.close()


