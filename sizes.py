import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

from halation import apply_halation_effect


def compute_adaptive_parameters(image):
    """
    Compute adaptive filter parameters based on image size and histogram.

    Parameters:
    - image: Input image (BGR format)

    Returns:
    - Dictionary of adaptive parameters
    """
    # Get image dimensions
    height, width = image.shape[:2]
    area = width * height

    # Reference area (medium image: 800x533 ≈ 426,400 pixels)
    ref_area = 800 * 533
    area_scale = np.sqrt(area / ref_area)  # Scale parameters with square root of area

    # Base parameters (for medium image)
    base_params = {
        "alpha": 0.5,
        "blur_radius": 15,
        "color_scale": 0.7,
        "canny_low": 100,
        "canny_high": 200,
        "dilation_size": 3,
        "s_curve_strength": 1.0,
        "grain_strength": 0.02,
    }

    # Adjust parameters based on image size
    adaptive_params = base_params.copy()
    adaptive_params["blur_radius"] = max(3, int(15 * area_scale) | 1)  # Ensure odd
    adaptive_params["dilation_size"] = max(1, int(3 * area_scale))
    adaptive_params["grain_strength"] = np.clip(0.02 * area_scale, 0.01, 0.05)

    # # Compute luminance histogram (Y channel in YUV)
    # yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # hist = cv2.calcHist([yuv_image], [0], None, [256], [0, 256])
    # hist = hist / hist.sum()  # Normalize to probabilities

    # # Check proportion of highlight pixels (luminance > 200)
    # highlight_ratio = hist[200:].sum()
    # if highlight_ratio > 0.2:  # If >20% highlights, tone down halation
    #     adaptive_params["alpha"] = adaptive_params["alpha"] * 0.6  # Reduce by 40%
    #     adaptive_params["color_scale"] = (
    #         adaptive_params["color_scale"] * 0.7
    #     )  # Reduce by 30%

    return adaptive_params


# Create images of different sizes
sizes = [
    {
        "name": "Original",
        "url": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YmlyZHxlbnwwfHwwfHx8Mg%3D%3D",
    },
    {
        "name": "Big",
        "url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8cG9ydHJhaXR8ZW58MHx8MHx8fDA%3D",
    },
    {
        "name": "Medium",
        "url": "https://images.pexels.com/photos/39866/entrepreneur-startup-start-up-man-39866.jpeg?cs=srgb&dl=pexels-pixabay-39866.jpg&fm=jpg",
    },
    {
        "name": "Small",
        "url": "https://i.ytimg.com/vi/CFfP76b0thM/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCRWs0bpsQyu8X-0TRgs9T1MmaaVw",
    },
]

default_params = {
    "alpha": 0.5,
    "blur_radius": 15,
    "color_scale": 0.7,
    "canny_low": 100,
    "canny_high": 200,
    "dilation_size": 3,
    "s_curve_strength": 1.0,
    "grain_strength": 0.02,
}

images = []
adaptive_params_list = []
for size_info in sizes:
    # Resize image to specified size
    response = requests.get(size_info["url"])
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    images.append(image)
    # Compute adaptive parameters for this size
    params = compute_adaptive_parameters(image)
    params["name"] = size_info["name"]
    adaptive_params_list.append(params)

# Define trackbar default values


# Display results for all images in a single window
plt.figure(figsize=(15, 10))

for i, (img, adaptive_params) in enumerate(zip(images, adaptive_params_list), 1):
    # Combine adaptive defaults with trackbar overrides
    params = {key: adaptive_params[key] for key in default_params}

    # Apply the halation effect
    result, _, _ = apply_halation_effect(
        img,
        alpha=params["alpha"],
        blur_radius=params["blur_radius"],
        color_scale=params["color_scale"],
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        dilation_size=params["dilation_size"],
        s_curve_strength=params["s_curve_strength"],
        grain_strength=params["grain_strength"],
    )

    # Display the result
    plt.subplot(2, 2, i)
    plt.title(adaptive_params["name"])
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()
