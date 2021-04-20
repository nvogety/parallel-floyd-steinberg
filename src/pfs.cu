#include <stdio.h>
#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>



#define BLOCKDIMCOMP 32
#define BLOCKSIZE 1024 // BLOCKDIMCOMP^2
#define FACTOR 4

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
__global__ void kernelPfs(int imageWidth, int imageHeight) {

    //TODO: What work needs to be done and how is work split?  
    
    //TODOL Some parallelization on sequential code
    


    // for(int y = 0; y < height-1; y++){
    //     for(int x = 1; x < width-1; x++){

                // TODO: Goal to make this part of the work parallel

    //         unsigned char* pixel = img + (x + width * y) * channels;
    //         unsigned char* pixel_right = img + ((x+1) + width * y) * channels;
    //         unsigned char* pixel_bottom_left = img + ((x-1) + width * (y+1)) * channels;
    //         unsigned char* pixel_bottom = img + (x + width * (y+1)) * channels;
    //         unsigned char* pixel_bottom_right = img + ((x+1) + width * (y+1)) * channels;

    //         int oldR = static_cast<int>(pixel[0]);
    //         int oldG = static_cast<int>(pixel[1]);
    //         int oldB = static_cast<int>(pixel[1]);

    //         int newR = round(FACTOR * oldR / 255.0) * (255/FACTOR);
    //         int newG = round(FACTOR * oldG / 255.0) * (255/FACTOR);
    //         int newB = round(FACTOR * oldB / 255.0) * (255/FACTOR);

    //         pixel[0] = newR;
    //         pixel[1] = newG;
    //         pixel[2] = newB;

    //         int qErrorR = oldR - newR;
    //         int qErrorG = oldG - newG;
    //         int qErrorB = oldB - newB;

    //         pixel_right[0] = (uint8_t)(pixel_right[0] + (qErrorR * (7.0 / 16.0)));
    //         pixel_right[1] = (uint8_t)(pixel_right[1] + (qErrorG * (7.0 / 16.0)));
    //         pixel_right[2] = (uint8_t)(pixel_right[2] + (qErrorB * (7.0 / 16.0)));

    //         pixel_bottom_left[0] = (uint8_t)(pixel_bottom_left[0] + (qErrorR * (3.0 / 16.0)));
    //         pixel_bottom_left[1] = (uint8_t)(pixel_bottom_left[1] + (qErrorG * (3.0 / 16.0)));
    //         pixel_bottom_left[2] = (uint8_t)(pixel_bottom_left[2] + (qErrorB * (3.0 / 16.0)));

    //         pixel_bottom[0] = (uint8_t)(pixel_bottom[0] + (qErrorR * (5.0 / 16.0)));
    //         pixel_bottom[1] = (uint8_t)(pixel_bottom[1] + (qErrorG * (5.0 / 16.0)));
    //         pixel_bottom[2] = (uint8_t)(pixel_bottom[2] + (qErrorB * (5.0 / 16.0)));

    //         pixel_bottom_right[0] = (uint8_t)(pixel_bottom_right[0] + (qErrorR * (1.0 / 16.0)));
    //         pixel_bottom_right[1] = (uint8_t)(pixel_bottom_right[1] + (qErrorG * (1.0 / 16.0)));
    //         pixel_bottom_right[2] = (uint8_t)(pixel_bottom_right[2] + (qErrorB * (1.0 / 16.0)));
            
    //         // cout << "R: " << (int)oldR << "->" << newR << " G: " << (int)oldG << "->" << newG << " B: " << (int)oldB << "->" << newB<< endl;
    //     }
    // }
    
    // cout << "Generated Color Dithered image" << endl; 

    // stbi_write_png("../images/dither.png", width, height, channels, img, width * channels);

    // cout << "Wrote Color Dithered image" << endl; 

    // return;

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

    // run kernel
    kernelPfs<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
   
    unsigned char *outputImageData;
    // copy result from GPU using cudaMemcpy: Idea is to copy computed image to local
    cudaMemcpy(outputImageData, cudaDeviceImageData, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

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