import argparse
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Load model and helper functions
def load_model(checkpoint_path, device):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", 
        encoder_weights="imagenet",
        in_channels=3, 
        classes=3
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def mask_to_rgb(mask, color_dict):
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for k in color_dict.keys():
        rgb_image[mask==k] = color_dict[k]
    return rgb_image

def preprocess_image(image_path, size):
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_w = img.shape[0]
    ori_h = img.shape[1]
    img = cv2.resize(img, (size, size))
    transformed = val_transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    return input_img, ori_h, ori_w

def main(image_path, output_path, checkpoint_path, device):
    # Load model
    model = load_model(checkpoint_path, device)

    # Preprocess image
    input_img, ori_h, ori_w = preprocess_image(image_path, size=256)
    input_img = input_img.to(device)

    # Inference
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Post-process mask
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}  # Example color map
    mask_rgb = mask_to_rgb(mask, color_dict)

    # Save output
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, mask_rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="segmented_output.png", help="Path to save the output image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join("checkpoints", "colorization_model.pth")

    main(args.image_path, args.output_path, checkpoint_path, device)
