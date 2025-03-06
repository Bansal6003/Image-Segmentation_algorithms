import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import os
import cv2


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(model_path):
    print("Loading model...")
    
    # Initialize DeepLabV3+ model for binary segmentation
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=1,  # Binary segmentation
    )
    
    # Load the trained weights
    checkpoint = torch.load(model_path)
    
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading from model_state_dict")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading state dict directly")
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    # Load grayscale image
    image = Image.open(image_path).convert('L')
    original_size = image.size
    
    # Create transform for grayscale image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Transform and prepare input
    input_tensor = transform(image)
    input_tensor = input_tensor.repeat(3, 1, 1)  # Convert to 3 channels
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return input_tensor, image, original_size

def predict(model, input_tensor, original_size):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass
        output = model(input_tensor)
        
        # Apply sigmoid for binary prediction
        probabilities = torch.sigmoid(output)
        
        print(f"Probability range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")
        
        # Get binary prediction with threshold
        prediction = (probabilities > 0.5).float().squeeze().cpu().numpy()
        
        # Print prediction statistics
        print(f"Prediction shape: {prediction.shape}")
        print(f"Foreground pixels: {np.sum(prediction == 1)} ({(np.sum(prediction == 1)/prediction.size)*100:.2f}%)")
        
        # Resize to original size
        prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))
        prediction_img = prediction_img.resize(original_size, Image.NEAREST)
        prediction = np.array(prediction_img) > 127
        
    return prediction.astype(np.uint8)

def calculate_area(prediction, pixel_size):
    """Calculate area of the segmented region in mm²"""
    # Count white pixels
    num_pixels = np.sum(prediction > 0)
    # Convert to mm²
    area_mm2 = num_pixels * (pixel_size ** 2)
    return area_mm2

def visualize_results(original_image, prediction, save_path=None, pixel_size=0.01):
    # Convert to numpy array if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Convert grayscale to RGB if needed
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Calculate area
    area_mm2 = calculate_area(prediction, pixel_size)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    # Plot binary mask with area
    plt.subplot(132)
    plt.title(f"Segmentation Mask\nArea: {area_mm2:.2f} mm²")
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    
    # Plot overlay with area
    plt.subplot(133)
    plt.title("Overlay")
    
    # Create red overlay for segmented regions
    overlay = np.zeros_like(original_image)
    overlay[prediction > 0] = [255, 0, 0]  # Red for segmented regions
    
    # Blend original and overlay
    alpha = 0.35  # Adjust transparency
    blended = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
    
    # Add area text to the image
    blended = blended.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Area: {area_mm2:.2f} mm2"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    
    # Position text at top-left corner with padding
    x = 10
    y = text_size[1] + 10
    
    # Add white background for text
    cv2.rectangle(blended, 
                 (x-5, y-text_size[1]-5),
                 (x+text_size[0]+5, y+5),
                 (255, 255, 255),
                 -1)
    
    # Add text
    cv2.putText(blended, text, (x, y), font, 1, (0, 0, 0), 2)
    
    plt.imshow(blended)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    else:
        plt.show()
        
    return area_mm2

def save_prediction(prediction, save_path, pixel_size=0.01):
    # Calculate area
    area_mm2 = calculate_area(prediction, pixel_size)
    
    # Save binary mask
    mask_image = Image.fromarray((prediction * 255).astype(np.uint8))
    mask_image.save(save_path)
    print(f"Saved binary mask to {save_path}")
    
    # Save colored visualization with area information
    colored_path = save_path.replace('_mask.png', '_colored_mask.png')
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    colored_mask[prediction > 0] = [255, 0, 0]  # Red for segmented regions
    
    # Add area text to colored mask
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Area: {area_mm2:.2f} mm2"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    
    # Position text at top-left corner with padding
    x = 10
    y = text_size[1] + 10
    
    # Add white background for text
    cv2.rectangle(colored_mask, 
                 (x-5, y-text_size[1]-5),
                 (x+text_size[0]+5, y+5),
                 (255, 255, 255),
                 -1)
    
    # Add text
    cv2.putText(colored_mask, text, (x, y), font, 1, (0, 0, 0), 2)
    
    Image.fromarray(colored_mask).save(colored_path)
    print(f"Saved colored mask to {colored_path} (Area: {area_mm2:.2f} mm²)")
    
    return area_mm2

def process_directory(model, input_dir, output_dir, pixel_size=0.01):
    # Dictionary to store results
    results = {}
    
    # Process all images in directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Setup paths
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            
            print(f"\nProcessing {filename}...")
            
            try:
                # Process image
                input_tensor, original_image, original_size = preprocess_image(input_path)
                prediction = predict(model, input_tensor, original_size)
                
                # Save results and get area
                area = save_prediction(prediction, mask_path, pixel_size)
                visualize_results(original_image, prediction, vis_path, pixel_size)
                
                # Store results
                results[filename] = {
                    'area_mm2': area,
                    'mask_path': mask_path,
                    'visualization_path': vis_path
                }
                
                print(f"Successfully processed {filename} (Area: {area:.2f} mm²)")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, 'area_measurements.csv'))
    print(f"\nSaved area measurements to {os.path.join(output_dir, 'area_measurements.csv')}")

def main():
    # Enable debug mode
    torch.set_printoptions(precision=10)
    
    # Paths
    model_path = r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\Deep_Lab\Trained_Models\Whole_Larva_deeplabv3plus_final_model.pth"
    input_dir = r"C:\Users\Pushkar Bansal\Desktop\Eff_Net_test_images"
    output_dir = r"D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\predictions_deeplabv3plus"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        print("Loading DeepLabV3+ model...")
        model = load_model(model_path)
        
        # Process directory
        print("Processing images...")
        pixel_size = 0.01  # mm per pixel
        process_directory(model, input_dir, output_dir, pixel_size=pixel_size)
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()