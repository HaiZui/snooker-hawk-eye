# Snooker Hawk-Eye Application

## Overview
The Snooker Hawk-Eye application is a Python-based tool that generates a 3D view of a snooker table from a 2D image. It allows users to analyze the positions of all balls on the table and determine if the white ball can hit other balls. The application provides a rotatable 3D view for better analysis.

## Features
- Capture a 2D image of the snooker table from the screen.
- Detect and track the positions of all balls on the table.
- Generate a 3D visualization of the table and balls.
- Rotate the camera freely to analyze ball positions and angles.

## Directory Structure
```
snooker-hawk-eye/
├── src/
│   ├── main.py                # Entry point of the application
│   ├── config.py              # Configuration settings
│   ├── image_processing/      # Image processing utilities
│   ├── ball_tracking/         # Ball detection and tracking
│   ├── table_detection/       # Table detection and calibration
│   ├── visualization/         # 3D rendering and camera control
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests for the application
├── data/                      # Sample images and models
├── requirements.txt           # Python dependencies
├── setup.py                   # Installation script
└── README.md                  # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/snooker-hawk-eye.git
   cd snooker-hawk-eye
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python src/main.py
   ```

2. Follow the on-screen instructions to capture a 2D image and generate a 3D view.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

