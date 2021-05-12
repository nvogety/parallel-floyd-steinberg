// Note: this file outlines a few unsuccesful attempts at getting a CUDA version of the algorithm to work
// but failed to produce speedup. Check out ./main.cpp for the OMP version and /mpi for the MPI version

#include <stdio.h>
#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "CycleTimer.h"

#define BLOCKDIMCOMP 4
#define FACTOR 4

#define TOP_RIGHT_ERR_DIFF (7.0 / 16.0)
#define BOT_LEFT_ERR_DIFF (3.0 / 16.0)
#define BOT_MID_ERR_DIFF (5.0 / 16.0)
#define BOT_RIGHT_ERR_DIFF (1.0 / 16.0)


// #define SCAN_BLOCK_DIM 1024 // BLOCKSIZE 
// #define NUMOFCIRCLESPERBLOCK 1750 // can change this
// #define NUMOFCIRCLESPERTHREAD 100 // NUMOFCIRCLESPERBLOCK/BLOCKSIZE rounded up

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////


struct GlobalConstants {
    int imageWidth;
    int imageHeight;
    int imageChannels;
    uint8_t* imageData;
};

__constant__ GlobalConstants cuConstRendererParams;

// Shareed memory attempt - memory caps out! Do not use!
__global__ void kernelPfsBlockFaster(unsigned char* img, int width, int height, int channels, int xblock, int yblock) {
    // Have all threads in block create a shared mem copy of img
    
    __shared__ unsigned char shared_img[20000];


    int threadID =  threadIdx.y * blockDim.x + threadIdx.x;
    int numOfThreads = blockDim.x * blockDim.y;

    printf("Thread %d\n", threadID); 
    
    int bytes = width * height * channels;
    int numPixPerThread = (bytes + numOfThreads - 1) / numOfThreads;
    int ind = threadID*numPixPerThread;
    __syncthreads();

    for(int k = 0; k < numPixPerThread && (ind + k) < bytes; k++){
        int pixInd = ind + k;
        shared_img[pixInd] = img[pixInd];
    }

    __syncthreads();
    if(threadID == 0){
        printf(" print finished transfer\n");
    }


    int blockMinX = threadIdx.x * xblock;
    int blockMaxX = blockMinX + xblock;
    int blockMinY = threadIdx.y * yblock;
    int blockMaxY = blockMinY + yblock;

    

    blockMinX = blockMinX == 0 ? 1 : blockMinX;

    for(int y = 0; y < blockMaxY && y < height-1; y++){
        for(int x = 1; x < blockMaxX && x < width-1; x++){

            unsigned char* pixel = shared_img + (x + width * y) * channels;
            unsigned char* pixel_right = shared_img + ((x+1) + width * y) * channels;
            unsigned char* pixel_bottom_left = shared_img + ((x-1) + width * (y+1)) * channels;
            unsigned char* pixel_bottom = shared_img + (x + width * (y+1)) * channels;
            unsigned char* pixel_bottom_right = shared_img + ((x+1) + width * (y+1)) * channels;

            int oldR = static_cast<int>(pixel[0]);
            int oldG = static_cast<int>(pixel[1]);
            int oldB = static_cast<int>(pixel[2]);

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
        }
    }

    return;
}


// Original with no shared memory - Very slow!
__global__ void kernelPfsBlock(unsigned char* img, int width, int height, int channels, int xblock, int yblock) {

    int blockMinX = threadIdx.x * xblock;
    int blockMaxX = blockMinX + xblock;
    int blockMinY = threadIdx.y * yblock;
    int blockMaxY = blockMinY + yblock;

    int threadID =  threadIdx.y * blockDim.x + threadIdx.x;

    blockMinX = blockMinX == 0 ? 1 : blockMinX;

    for(int y = 0; y < blockMaxY && y < height-1; y++){
        for(int x = 1; x < blockMaxX && x < width-1; x++){

            unsigned char* pixel = img + (x + width * y) * channels;
            unsigned char* pixel_right = img + ((x+1) + width * y) * channels;
            unsigned char* pixel_bottom_left = img + ((x-1) + width * (y+1)) * channels;
            unsigned char* pixel_bottom = img + (x + width * (y+1)) * channels;
            unsigned char* pixel_bottom_right = img + ((x+1) + width * (y+1)) * channels;

            int oldR = static_cast<int>(pixel[0]);
            int oldG = static_cast<int>(pixel[1]);
            int oldB = static_cast<int>(pixel[2]);

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
        }
    }

    return;
}

// Trying with global params, like A2 - very similar performance!
__global__ void kernelPfsBlockParams(int width, int height, int channels, int xblock, int yblock) {

    int blockMinX = threadIdx.x * xblock;
    int blockMaxX = blockMinX + xblock;
    int blockMinY = threadIdx.y * yblock;
    int blockMaxY = blockMinY + yblock;

    int threadID =  threadIdx.y * blockDim.x + threadIdx.x;

    blockMinX = blockMinX == 0 ? 1 : blockMinX;

    for(int y = 0; y < blockMaxY && y < height-1; y++){
        for(int x = 1; x < blockMaxX && x < width-1; x++){

            unsigned char* pixel = cuConstRendererParams.imageData + (x + width * y) * channels;
            unsigned char* pixel_right = cuConstRendererParams.imageData + ((x+1) + width * y) * channels;
            unsigned char* pixel_bottom_left = cuConstRendererParams.imageData + ((x-1) + width * (y+1)) * channels;
            unsigned char* pixel_bottom = cuConstRendererParams.imageData + (x + width * (y+1)) * channels;
            unsigned char* pixel_bottom_right = cuConstRendererParams.imageData + ((x+1) + width * (y+1)) * channels;

            int oldR = static_cast<int>(pixel[0]);
            int oldG = static_cast<int>(pixel[1]);
            int oldB = static_cast<int>(pixel[2]);

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
        }
    }

    return;
}

void pfsCuda(int width, int height, int channels, unsigned char *img) {

    printf(" In CUDA code\n");

    dim3 blockDim(BLOCKDIMCOMP, BLOCKDIMCOMP);
    dim3 gridDim(1, 1);

    int xblock = (width + blockDim.x - 1) / blockDim.x;
    int yblock = (height + blockDim.y - 1) / blockDim.y;

    int bytes = sizeof(uint8_t) * width * height * channels;

    unsigned char *device_img;
    unsigned char *result = new unsigned char[width * height * channels];


    unsigned char *cudaDeviceImageData;

    cudaMalloc(&cudaDeviceImageData, bytes);

    cudaMemcpy(cudaDeviceImageData, img, bytes, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.imageWidth = width;
    params.imageHeight = height;
    params.imageChannels = channels;
    params.imageData = cudaDeviceImageData;
    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    cudaMalloc(&device_img, bytes);
    cudaMemcpy(device_img, img, bytes, cudaMemcpyHostToDevice);

    printf(" Arrays Copied\n");
    printf(" Starting kernel\n");

    double startKernelTime = CycleTimer::currentSeconds();

    // run kernel
    // Change this to try our three different approaches
    kernelPfsBlockParams<<<gridDim, blockDim>>>(width, height, channels, xblock, yblock);

    cudaDeviceSynchronize();
    double endKernelTime = CycleTimer::currentSeconds();

    printf(" CUDA finished\n");
    double kernelDuration = endKernelTime - startKernelTime;
    printf(" Kernel Time: %f\n", kernelDuration);
   
    cudaMemcpy(result, cudaDeviceImageData, bytes, cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    printf(" Generated Color Dithered image\n"); 

    stbi_write_png("../images/dither-cuda.png", width, height, channels, result, width * channels);
    printf(" Wrote Color Dithered image\n"); 

    cudaFree(device_img);

    cudaFree(cudaDeviceImageData);
}

int cudaMain() {

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