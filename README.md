# Video Streaming App with CUDA

This application processes a video stream in real-time using a CUDA-accelerated edge detection filter.

## Requirements

- Python 3.6+
- CUDA toolkit
- OpenCV
- NumPy

## Installation

1. Build the CUDA code:
    ```
    nvcc -o src/cuda_filters.so --shared -Xcompiler -fPIC src/cuda_filters.cu
    ```

2. Install Python dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

Run the application:

```python
    python3 src/video_stream_processor.py
```
