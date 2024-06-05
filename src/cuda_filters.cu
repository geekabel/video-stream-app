#include <cuda_runtime.h>
#include <math.h>

extern "C" {
__global__ void edge_detection(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -input[(y-1) * width + (x-1)] - 2*input[y * width + (x-1)] - input[(y+1) * width + (x-1)]
                 + input[(y-1) * width + (x+1)] + 2*input[y * width + (x+1)] + input[(y+1) * width + (x+1)];
        int gy = -input[(y-1) * width + (x-1)] - 2*input[(y-1) * width + x] - input[(y-1) * width + (x+1)]
                 + input[(y+1) * width + (x-1)] + 2*input[(y+1) * width + x] + input[(y+1) * width + (x+1)];
        output[idx] = min(255, max(0, (int)sqrtf(gx * gx + gy * gy)));
    }
}

void edgeDetection(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char* d_input;
    unsigned char* d_output;
    int size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    edge_detection<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
}
