# BKAI-IGH NeoPolyp Segmentation Assignment

This repository contains the code and resources for the BKAI-IGH NeoPolyp Segmentation Assignment. Follow the instructions below to set up the repository and run the project.

## Checkpoints

The model checkpoints are available for download via the following Google Drive link:  
[Google Drive: Checkpoints](https://drive.google.com/drive/folders/1LROAdLQM0xn5NqGa3lW4XpuW4XHxSAms?usp=drive_link)

### Directory Structure

After downloading the `checkpoints` folder, your repository should be structured as follows:

<BKAI-IGH NeoPolyp Segmentation Assignment>/ ├── checkpoints/ # Directory for saved model checkpoints │ └── colorization_model.pth ├── infer.py # Inference script ├── README.md # Project documentation ├── dl-assignment-segmentation.ipynb # Notebook for training ├── requirements.txt # Dependencies for the project

## Requirements

Make sure to install the dependencies listed in `requirements.txt` before running the code:

pip install -r requirements.txt

## Inference

Run the following command to perform inference:

python infer.py --image_path <input_image_path> --output_path <output_image_path>
Replace <input_image_path> with the path to the input image and <output_image_path> with the desired output path for the segmented image.

## Training

The training notebook is available as dl-assignment-segmentation.ipynb. Open it in a Jupyter Notebook environment and follow the instructions inside to train the model.
