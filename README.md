# CS2 Analyzer Web Application

## Project Overview
This project is a web application built using Flask, OpenCV, and EasyOCR to analyze CS2 (Counter-Strike 2) gameplay images. The application focuses on detecting the map, extracting information from the scoreboard, and processing relevant text data using Optical Character Recognition (OCR).

## Key Features
- **Map Detection:** Analyze gameplay images and classify the map using histogram comparison techniques.
- **Scoreboard Analysis:** Automatically crop and process scoreboard regions for better text recognition.
- **OCR Integration:** Extract text from images using EasyOCR with optional GPU support for faster processing.
- **Debug Mode:** Save and visualize intermediate steps for easier debugging.

## Dependencies
To run this project, ensure the following dependencies are installed:

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- EasyOCR (`easyocr`)
- Flask (`flask`)
- CUDA (optional for GPU acceleration with EasyOCR)

Install the dependencies using:
```bash
pip install opencv-python numpy easyocr flask
```

## Project Structure
```
CS2-Analyzer-v1.0/
├── app.py               # Main Flask application
├── static/
│   └── debug_cs2_project # Directory for debug images
├── templates/            # HTML templates for the web interface
└── dataset/              # Dataset containing map reference images
    ├── ancient/
    ├── nuke/
    └── anubis/
```

## Configuration
**Global Parameters:**
- `DEBUG`: Enable or disable debug mode.
- `DEBUG_DIR`: Directory to store debug images.
- `CONFIDENCE_THRESHOLD`: Minimum OCR confidence level to accept text.
- `BINS`: Number of histogram bins for map detection.

### Scoreboard Region Parameters
- `ROI_TOP`, `ROI_BOTTOM`, `ROI_LEFT`, `ROI_RIGHT`: Define the region of interest for the scoreboard.
- `CROP_SIDE_LEFT`, `CROP_SIDE_RIGHT`: Pixel values for cropping the sides of the scoreboard.
- `KEEP_TOP_HALF`: Boolean flag to keep only the top half of the scoreboard.

## How to Run the Application
1. **Prepare the Dataset:**
   Ensure that representative images for each map (Ancient, Nuke, Anubis, etc.) are placed in the corresponding folders under the `dataset/` directory.

2. **Run the Flask App:**
   ```bash
   python app.py
   ```

3. **Access the Application:**
   Open your web browser and navigate to `http://127.0.0.1:5000`

## Usage Instructions
1. **Upload Image:** Upload a CS2 gameplay screenshot.
2. **Analyze Image:** The system will detect the map, extract scoreboard information, and display OCR results.
3. **View Debug Images:** Debug images can be viewed in the `static/debug_cs2_project` directory.

## Debugging Functions
- `ensure_debug_dir()`: Create the debug directory if it does not exist.
- `debug_save(filename, img)`: Save intermediate images for debugging.

## OCR and Preprocessing
- Various preprocessing techniques are applied, such as grayscale conversion, resizing, and thresholding.
- OCR is performed using EasyOCR with optional text filtering.

## Map Detection
- Histogram-based classification is used to match gameplay images with reference map images.
- Map signatures are precomputed for better performance.

## Notes
- **CUDA Requirement:** For GPU acceleration, ensure that NVIDIA CUDA is installed and configured.
- **File Paths:** Update the file paths and dataset locations as per your local setup.

## Example Output
```
=== FINAL RESULTS ===
Timer: 1:30, CT Score: 10, T Score: 12, Map: Ancient
CT Economy: 5000, T Economy: 6000
```

## Contributing
We welcome contributions to improve the functionality, performance, and usability of the application. Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Commit your changes with clear messages.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Future Improvements
- **User Authentication:** Allow user accounts for better data management.
- **Real-Time Analysis:** Support for processing live gameplay feeds.
- **Enhanced Map Detection:** Utilize deep learning models for more accurate map recognition.
- **Performance Optimization:** Improve OCR and image processing speed.

