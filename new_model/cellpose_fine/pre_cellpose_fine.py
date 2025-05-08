import os
import argparse
import numpy as np
import cv2
from cellpose import models, io


def process_images(model_path, input_dir, output_dir, diameter=0, flow_threshold=0.4, cellprob_threshold=0):
    """Main processing function to run Cellpose model and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # Use model diameter if diameter is 0
    diameter = model.diam_labels if diameter == 0 else diameter

    # Get input files and images
    files = io.get_image_files(input_dir, mask_filter='_mask')
    images = [io.imread(f) for f in files]

    # Run model
    masks, flows, styles = model.eval(
        images,
        channels=[0, 0],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )

    # Save masks
    for i in range(len(masks)):
        mask = masks[i]
        file = os.path.basename(files[i])
        save_path = os.path.join(output_dir, file)

        save_mask = np.zeros_like(mask).astype(np.uint8)
        save_mask[mask > 0] = 255

        print(f"Saving: {save_path}")
        cv2.imwrite(save_path, save_mask)


def main():
    """Entry point of the script"""
    args = parse_arguments()
    process_images(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Cellpose model on input images.')
    parser.add_argument('-m','--model_path', required=True, help='Path to the Cellpose model')
    parser.add_argument('-i','--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('-o','--output_dir', required=True, help='Directory to save output masks')
    parser.add_argument('--diameter', type=float, default=0, help='Cell diameter (0 for automatic)')
    parser.add_argument('--flow_threshold', type=float, default=0.4, help='Flow threshold')
    parser.add_argument('--cellprob_threshold', type=float, default=0, help='Cell probability threshold')

    return parser.parse_args()

if __name__ == '__main__':
    main()