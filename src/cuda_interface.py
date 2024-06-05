import ctypes
import numpy as np
import os

# Load the shared library
cuda_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'cuda_filters.so'))

# Define the function signature
cuda_lib.edgeDetection.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]

def edge_detection(input_image: np.ndarray) -> np.ndarray:
    height, width = input_image.shape
    output_image = np.zeros_like(input_image)

    input_ptr = input_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    output_ptr = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    cuda_lib.edgeDetection(input_ptr, output_ptr, width, height)

    return output_image
