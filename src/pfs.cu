#include <stdio.h>
#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>



#define BLOCKDIMCOMP 32
#define BLOCKSIZE 1024 // BLOCKDIMCOMP^2
#define FACTOR 4

#define TOP_RIGHT_ERR_DIFF (7 / 16)
#define BOT_LEFT_ERR_DIFF (3 / 16)
#define BOT_MID_ERR_DIFF (5 / 16)
#define BOT_RIGHT_ERR_DIFF (1 / 16)


// #define SCAN_BLOCK_DIM 1024 // BLOCKSIZE 
// #define NUMOFCIRCLESPERBLOCK 1750 // can change this
// #define NUMOFCIRCLESPERTHREAD 100 // NUMOFCIRCLESPERBLOCK/BLOCKSIZE rounded up

#include <cuda.h>

#include <cuda_runtime.h>
#include <driver_functions.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////


// kernelPfs -- (CUDA device code)
// TODO: Specific work each cuda block work still needs to be divided.
__global__ void kernelPfs(int imageWidth, int imageHeight, int channels, unsigned char* img, unsigned char* output) {

    //TODO: What work needs to be done and how is work split?  
    
    // Boundary of box we are computing -> Don't know if this is useful but for conceptual understanding
    int blockMinX = blockIdx.x * blockDim.x;
    int blockMaxX = blockMinX + blockDim.x;
    int blockMinY = blockIdx.y * blockDim.y;
    int blockMaxY = blockMinY + blockDim.y;

    int threadID =  threadIdx.y * blockDim.x + threadIdx.x;

    unsigned char* pixel = img + (blockIdx.x + imageWidth * blockIdxy) * channels;
    unsigned char* pixel_right = img + ((blockIdx.x+1) + imageWidth * blockIdx.y) * channels;
    unsigned char* pixel_bottom_left = img + ((blockIdx.x-1) + imageWidth * (blockIdx.y+1)) * channels;
    unsigned char* pixel_bottom = img + (blockIdx.x + width * (blockIdx.y+1)) * channels;
    unsigned char* pixel_bottom_right = img + ((blockIdx.y+1) + imageWidth * (blockIdx.y+1)) * channels;

    int oldR = static_cast<int>(pixel[0]);
    int oldG = static_cast<int>(pixel[1]);
    int oldB = static_cast<int>(pixel[1]);

    int newR = round(FACTOR * oldR / 255.0) * (255/FACTOR);
    int newG = round(FACTOR * oldG / 255.0) * (255/FACTOR);
    int newB = round(FACTOR * oldB / 255.0) * (255/FACTOR);

    pixel[0] = newR;
    pixel[1] = newG;
    pixel[2] = newB;

    int qErrorR = oldR - newR;
    int qErrorG = oldG - newG;
    int qErrorB = oldB - newB;

    pixel_right[0] = (uint8_t)(pixel_right[0] + (qErrorR * TOP_RIGHT_ERR_DIFF));
    pixel_right[1] = (uint8_t)(pixel_right[1] + (qErrorG * TOP_RIGHT_ERR_DIFF));
    pixel_right[2] = (uint8_t)(pixel_right[2] + (qErrorB * TOP_RIGHT_ERR_DIFF));

    pixel_bottom_left[0] = (uint8_t)(pixel_bottom_left[0] + (qErrorR * BOT_LEFT_ERR_DIFF));
    pixel_bottom_left[1] = (uint8_t)(pixel_bottom_left[1] + (qErrorG * BOT_LEFT_ERR_DIFF));
    pixel_bottom_left[2] = (uint8_t)(pixel_bottom_left[2] + (qErrorB * BOT_LEFT_ERR_DIFF));

    pixel_bottom[0] = (uint8_t)(pixel_bottom[0] + (qErrorR * BOT_MID_ERR_DIFF));
    pixel_bottom[1] = (uint8_t)(pixel_bottom[1] + (qErrorG * BOT_MID_ERR_DIFF));
    pixel_bottom[2] = (uint8_t)(pixel_bottom[2] + (qErrorB * BOT_MID_ERR_DIFF));

    pixel_bottom_right[0] = (uint8_t)(pixel_bottom_right[0] + (qErrorR * BOT_RIGHT_ERR_DIFF));
    pixel_bottom_right[1] = (uint8_t)(pixel_bottom_right[1] + (qErrorG * BOT_RIGHT_ERR_DIFF));
    pixel_bottom_right[2] = (uint8_t)(pixel_bottom_right[2] + (qErrorB * BOT_RIGHT_ERR_DIFF));
    
    
    // idk how this write works - Is it just making the modification? So can we just partially 
    // write each part.
    

    return;

}

void pfsCuda(int width, int height, int channels, unsigned char *img) {

    // Define dimensions for work assignment
    dim3 blockDim(BLOCKDIMCOMP, BLOCKDIMCOMP);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    unsigned char *cudaDeviceImageData;

    // Allocate device memory buffers on the GPU using cudaMalloc
    //TODO: Slightly confused what needs to be malloced on device - not sure if this is correct
    cudaMalloc(&cudaDeviceImageData, sizeof(uint8_t) * width * height * channels);


    // TODO copy input arrays to the GPU using cudaMemcpy
    // cudaMemcpy( cudaMemcpyHostToDevice);
    unsigned char *outputImageData;

    // run kernel
    kernelPfs<<<gridDim, blockDim>>>(imageHeight, imageWidth, channels, img, outputImageData);

    // TODO: Uncomment and check for correctness that image still renders
    // kernelPfs<<<1, 1>>>();

    cudaDeviceSynchronize();
   
    unsigned char *outputImageData;
    // copy result from GPU using cudaMemcpy: Idea is to copy computed image to local
    cudaMemcpy(outputImageData, cudaDeviceImageData, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    cout << "Generated Color Dithered image" << endl; 
    stbi_write_png("../images/dither.png", imageWidth, imageHeight, channels, outputImageData, imageWidth * channels);
    cout << "Wrote Color Dithered image" << endl; 

    // free memory buffers on the GPU
    cudaFree(cudaDeviceImageData);

}

int main () {

    //Load image 
    int width, height, channels;
    unsigned char *img = stbi_load("../images/landscape.png", &width, &height, &channels, 0);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    printf("Read image, width: %d, height: %d, channels: %d", width, height, channels);

    pfsCuda(width, height, channels, img);
    stbi_image_free(img);

    return 0;
}


void printCudaInfo() {
    // for fun, just print out some stats on the machine
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}