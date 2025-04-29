import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

from halation import apply_halation_effect

# Fetch the image from the URL
url = 'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YmlyZHxlbnwwfHwwfHx8Mg%3D%3D'
response = requests.get(url)
image = np.asarray(bytearray(response.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Define parameter sets to demonstrate different effect intensities
parameter_sets = [
    {
        "alpha": 0.7, "blur_radius": 21, "color_scale": 1.0,
        "canny_low": 100, "canny_high": 200, "dilation_size": 3,
        "s_curve_strength": 1.2, "grain_strength": 0.03,
        "title": "Strong Film Effect"
    },
    {
        "alpha": 0.5, "blur_radius": 21, "color_scale": 0.7,
        "canny_low": 100, "canny_high": 200, "dilation_size": 3,
        "s_curve_strength": 1.0, "grain_strength": 0.02,
        "title": "Moderate Film Effect"
    },
    {
        "alpha": 0.5, "blur_radius": 11, "color_scale": 0.7,
        "canny_low": 150, "canny_high": 250, "dilation_size": 2,
        "s_curve_strength": 0.8, "grain_strength": 0.01,
        "title": "Subtle Film Effect"
    },
]

# Display results for different parameter sets
plt.figure(figsize=(15, 10))

for i, params in enumerate(parameter_sets, 1):
    # Apply the halation effect with S-curve and grain
    result, mask, edges = apply_halation_effect(
        image,
        alpha=params["alpha"],
        blur_radius=params["blur_radius"],
        color_scale=params["color_scale"],
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        dilation_size=params["dilation_size"],
        s_curve_strength=params["s_curve_strength"],
        grain_strength=params["grain_strength"]
    )

    # Display the result
    plt.subplot(3, 2, i * 2 - 1)
    plt.title(params["title"])
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Display the corresponding mask
    plt.subplot(3, 2, i * 2)
    plt.title(f"Mask ({params['title']})")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()