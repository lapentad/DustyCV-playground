"""Image processing module for halation effect."""
import cv2
import numpy as np

def apply_halation_effect(image, alpha=0.3, blur_radius=15, color_scale=0.5,
                         canny_low=100, canny_high=200, dilation_size=3,
                         s_curve_strength=1.0, grain_strength=0.02):
    """
    Apply a halation effect with S-curve and monochrome grain.

    Parameters:
    - image: Input image (BGR format)
    - alpha: Blending strength of the halation glow (0.0 to 1.0)
    - blur_radius: Size of the Gaussian blur kernel for the glow (must be odd)
    - color_scale: Scaling factor for the glow color intensity (0.0 to 1.0)
    - canny_low: Lower threshold for Canny edge detection
    - canny_high: Upper threshold for Canny edge detection
    - dilation_size: Size of the dilation kernel for edges
    - s_curve_strength: Strength of the S-curve effect (0.0 to 2.0)
    - grain_strength: Strength of the monochrome grain (0.0 to 0.1)

    Returns:
    - Processed image with halation, S-curve, and grain
    - Halation mask (for visualization)
    - Edges (for visualization)
    """
    # Apply S-curve for film-like contrast
    image = apply_s_curve(image, s_curve_strength)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image to smooth it
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred_image, canny_low, canny_high)

    # Dilate edges to make them thicker
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a mask for the halation effect by blurring the dilated edges
    blur_kernel_size = (blur_radius, blur_radius)
    halation_mask = cv2.GaussianBlur(dilated_edges, blur_kernel_size, blur_radius // 3 + 3)

    # Normalize the mask to [0, 1] for blending
    halation_mask = halation_mask / 255.0

    # Create a red-tinted glow with adjustable color intensity
    red_glow = np.zeros_like(image, dtype=np.float32)
    red_glow[:, :, 2] = halation_mask * 255 * color_scale  # Red channel
    red_glow[:, :, 1] = halation_mask * 50 * color_scale   # Slight green for warmth
    red_glow[:, :, 0] = halation_mask * 50 * color_scale   # Slight blue for warmth

    # Blend the red glow with the original image
    image_with_halation = cv2.addWeighted(image.astype(np.float32), 1.0, red_glow, alpha, 0.0)
    image_with_halation = np.clip(image_with_halation, 0, 255).astype(np.uint8)

    # Add monochrome grain
    image_with_halation = add_monochrome_grain(image_with_halation, grain_strength)

    return image_with_halation, halation_mask, edges

def apply_s_curve(image, strength=1.0):
    """
    Apply an S-curve tone mapping to the image for a film-like contrast.

    Parameters:
    - image: Input image (BGR format)
    - strength: Strength of the S-curve effect (0.0 to 2.0, 1.0 is standard)

    Returns:
    - Image with S-curve applied
    """
    # Create a lookup table for the S-curve using a sigmoid-like function
    x = np.linspace(0, 255, 256)
    # Sigmoid curve: y = 255 / (1 + exp(-a * (x - 128) / 255)), adjusted by strength
    a = 5.0 * strength  # Controls the steepness of the curve
    lut = 255 / (1 + np.exp(-a * (x - 128) / 255))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply the LUT to each channel
    return cv2.LUT(image, lut)

def add_monochrome_grain(image, grain_strength=0.02):
    """
    Add monochrome film grain to the image.

    Parameters:
    - image: Input image (BGR format)
    - grain_strength: Strength of the grain (0.0 to 0.1, 0.02 is subtle)

    Returns:
    - Image with monochrome grain
    """
    # Convert to YUV color space to work on luminance
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Generate random noise for the luminance channel
    noise = np.random.normal(0, 255 * grain_strength, yuv_image.shape[:2]).astype(np.float32)

    # Add noise to the Y channel (luminance)
    yuv_image[:, :, 0] = yuv_image[:, :, 0].astype(np.float32) + noise
    yuv_image[:, :, 0] = np.clip(yuv_image[:, :, 0], 0, 255).astype(np.uint8)

    # Convert back to BGR
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)