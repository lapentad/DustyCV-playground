import cv2
import requests
import numpy as np

from halation import apply_halation_effect

def nothing(x):
    """Callback function for trackbars (required by OpenCV)."""
    pass

# Fetch the image from the URL
#url = 'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YmlyZHxlbnwwfHwwfHx8Mg%3D%3D'
url = 'https://images.pexels.com/photos/39866/entrepreneur-startup-start-up-man-39866.jpeg?cs=srgb&dl=pexels-pixabay-39866.jpg&fm=jpg'
response = requests.get(url)
image = np.asarray(bytearray(response.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
max_width = 800
max_height = 800
height, width = image.shape[:2]

if width > max_width or height > max_height:
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Create a window for the GUI
window_name = 'Film Effect with Halation'
cv2.namedWindow(window_name)

# Create trackbars for each parameter
cv2.createTrackbar('Alpha x100', window_name, 50, 100, nothing)  # 0.0 to 1.0
cv2.createTrackbar('Blur Radius', window_name, 15, 50, nothing)  # 3 to 51 (odd)
cv2.createTrackbar('Color Scale x100', window_name, 70, 100, nothing)  # 0.0 to 1.0
cv2.createTrackbar('Canny Low', window_name, 100, 300, nothing)  # 0 to 300
cv2.createTrackbar('Canny High', window_name, 200, 300, nothing)  # 0 to 300
cv2.createTrackbar('Dilation Size', window_name, 3, 10, nothing)  # 1 to 10
cv2.createTrackbar('S-Curve x100', window_name, 100, 200, nothing)  # 0.0 to 2.0
cv2.createTrackbar('Grain x1000', window_name, 20, 100, nothing)  # 0.0 to 0.1

while True:
    # Get current trackbar positions
    alpha = cv2.getTrackbarPos('Alpha x100', window_name) / 100.0
    blur_radius = cv2.getTrackbarPos('Blur Radius', window_name)
    color_scale = cv2.getTrackbarPos('Color Scale x100', window_name) / 100.0
    canny_low = cv2.getTrackbarPos('Canny Low', window_name)
    canny_high = cv2.getTrackbarPos('Canny High', window_name)
    dilation_size = cv2.getTrackbarPos('Dilation Size', window_name)
    s_curve_strength = cv2.getTrackbarPos('S-Curve x100', window_name) / 100.0
    grain_strength = cv2.getTrackbarPos('Grain x1000', window_name) / 1000.0

    # Ensure blur_radius is odd and at least 3
    blur_radius = max(3, blur_radius | 1)

    # Ensure dilation_size is at least 1
    dilation_size = max(1, dilation_size)

    # Apply the halation effect with current parameters
    result, _, _ = apply_halation_effect(
        image,
        alpha=alpha,
        blur_radius=blur_radius,
        color_scale=color_scale,
        canny_low=canny_low,
        canny_high=canny_high,
        dilation_size=dilation_size,
        s_curve_strength=s_curve_strength,
        grain_strength=grain_strength
    )

    cv2.imshow(window_name, result)

    # Break the loop on 'q' key press
    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break