import argparse
import os
import glob
from PIL import Image
import torch

from incubator.HRNet.inference import init, inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retina_weights_path", "-r", dest="retina_weights_path", default=r"Resnet50_Final.pth", help="")
    parser.add_argument("--image_dir", "-i", dest="image_dir", default=r"crop/", help="read xxx.mat")
    parser.add_argument("--save_dir", "-s", dest="save_dir", default=r"crop/", help="folder path to save the images")

    parser.add_argument("--rf_model_path", "-rf", dest="rf_model_path", default="Resnet50_Final.pth")
    parser.add_argument("--lm_model_path", "-lm", dest="lm_model_path", default="HR18-WFLW.pth")

    opt = parser.parse_args()


    rf_detector, lm_model = init(opt.rf_model_path, opt.lm_model_path, 98)


    for image_path in glob.glob(os.path.join(opt.image_dir, "*.png")):
        point_txt_path = os.path.join(
            opt.save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")

        image = Image.open(image_path)
        try:
            landmarks = inference(rf_detector, lm_model, image, 98)
        except:
            print(f"No face: {image_path}")
            continue

        with open(point_txt_path, "w") as f:
            for point_index in [96, 97]:
                x, y = landmarks[point_index]
                f.write(f"{x}	{y}\n")

            # nose:
            x_53, y_53 = landmarks[53]
            x_54, y_54 = landmarks[54]
            f.write(f"{(x_53 + x_54) // 2}	{(x_53 + y_54) // 2}\n")

            for point_index in [76, 82]:
                x, y = landmarks[point_index]
                f.write(f"{x}	{y}\n")
        f.close()
