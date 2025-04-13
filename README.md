# Face Mask Detection

A real-time face mask detection system using OpenCV and a pre-trained deep learning model. This project captures video from a webcam or video file and detects whether individuals are wearing face masks or not.

## Features

- Real-time detection via webcam or video input
- Classifies faces as:
  - With Mask 😷
  - Without Mask 😐
- Visual feedback with bounding boxes and labels
- Uses a pre-trained model for accurate classification

## Requirements

Make sure the following Python packages are installed:

```bash
pip install opencv-python imutils numpy
```

> **Note:** The code assumes access to a `.caffemodel` and `.prototxt` file for face detection, and a pre-trained Keras/TensorFlow model for mask classification. You will need to supply these files.

## Usage

```bash
python detect_mask_video.py
```

- To run on the webcam, simply execute the script.
- To run on a video file, change the video source inside the script:
  ```python
  vs = cv2.VideoCapture("path_to_video.mp4")
  ```

## File Structure

- `detect_mask_video.py` — Main detection script

## How it works

1. Loads a face detector model (Caffe-based).
2. Loads a mask classifier model (TensorFlow/Keras).
3. Captures video frames.
4. Detects faces in each frame.
5. Classifies each detected face as "Mask" or "No Mask".
6. Displays results with labels and bounding boxes.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- Pre-trained models from publicly available datasets
