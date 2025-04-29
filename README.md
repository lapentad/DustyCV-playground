# OpenCV Halation Effect Project

This project demonstrates the application of a halation effect, S-curve tone mapping, and monochrome grain to images using OpenCV. The project includes two main scripts:

- `main.py`: Displays different parameter sets for the halation effect using Matplotlib.
- `sliders.py`: Provides a GUI with sliders to adjust parameters in real-time.

## Installation

1. Clone the repository or download the files.
2. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt

## Usage
Run the Parameter Demo
To see the halation effect with predefined parameter sets, run:

```
python main.py
```

Run the Interactive GUI
To adjust the halation effect parameters in real-time, run:

```
python sliders.py
```

## Requirements
The project requires the following Python packages:

matplotlib
numpy
opencv-python
requests
These will be installed automatically when you run the pip install command above.

## Features
Halation Effect: Adds a glowing effect around bright areas of the image.
S-Curve Tone Mapping: Enhances contrast for a film-like look.
Monochrome Grain: Adds subtle film grain for a vintage aesthetic.
Interactive GUI: Adjust parameters dynamically using sliders.
Example Image
The scripts fetch an example image from the web to demonstrate the effects. You can replace the URL in the scripts with your own image URL if desired.

## License
This project is licensed under the MIT License.