#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "CycleTimer.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

using namespace std;

void printCudaInfo();
void pfsCuda(int width, int height, int channels, unsigned char *img);

void color_dither(unsigned char *img, int width, int height, int channels, int factor){
    
    for(int y = 0; y < height-1; y++){
        for(int x = 1; x < width-1; x++){
            unsigned char* pixel = img + (x + width * y) * channels;
            unsigned char* pixel_right = img + ((x+1) + width * y) * channels;
            unsigned char* pixel_bottom_left = img + ((x-1) + width * (y+1)) * channels;
            unsigned char* pixel_bottom = img + (x + width * (y+1)) * channels;
            unsigned char* pixel_bottom_right = img + ((x+1) + width * (y+1)) * channels;

            int oldR = static_cast<int>(pixel[0]);
            int oldG = static_cast<int>(pixel[1]);
            int oldB = static_cast<int>(pixel[1]);

            int newR = round(factor * oldR / 255.0) * (255/factor);
            int newG = round(factor * oldG / 255.0) * (255/factor);
            int newB = round(factor * oldB / 255.0) * (255/factor);

            pixel[0] = newR;
            pixel[1] = newG;
            pixel[2] = newB;

            int qErrorR = oldR - newR;
            int qErrorG = oldG - newG;
            int qErrorB = oldB - newB;

            pixel_right[0] = (uint8_t)(pixel_right[0] + (qErrorR * (7.0 / 16.0)));
            pixel_right[1] = (uint8_t)(pixel_right[1] + (qErrorG * (7.0 / 16.0)));
            pixel_right[2] = (uint8_t)(pixel_right[2] + (qErrorB * (7.0 / 16.0)));

            pixel_bottom_left[0] = (uint8_t)(pixel_bottom_left[0] + (qErrorR * (3.0 / 16.0)));
            pixel_bottom_left[1] = (uint8_t)(pixel_bottom_left[1] + (qErrorG * (3.0 / 16.0)));
            pixel_bottom_left[2] = (uint8_t)(pixel_bottom_left[2] + (qErrorB * (3.0 / 16.0)));

            pixel_bottom[0] = (uint8_t)(pixel_bottom[0] + (qErrorR * (5.0 / 16.0)));
            pixel_bottom[1] = (uint8_t)(pixel_bottom[1] + (qErrorG * (5.0 / 16.0)));
            pixel_bottom[2] = (uint8_t)(pixel_bottom[2] + (qErrorB * (5.0 / 16.0)));

            pixel_bottom_right[0] = (uint8_t)(pixel_bottom_right[0] + (qErrorR * (1.0 / 16.0)));
            pixel_bottom_right[1] = (uint8_t)(pixel_bottom_right[1] + (qErrorG * (1.0 / 16.0)));
            pixel_bottom_right[2] = (uint8_t)(pixel_bottom_right[2] + (qErrorB * (1.0 / 16.0)));
            
            // cout << "R: " << (int)oldR << "->" << newR << " G: " << (int)oldG << "->" << newG << " B: " << (int)oldB << "->" << newB<< endl;
        }
    }
    
    cout << "Generated Color Dithered image" << endl; 

    stbi_write_png("../images/dither-seq.png", width, height, channels, img, width * channels);

    cout << "Wrote Color Dithered image" << endl; 

    return;
}

void gray(unsigned char *img, int width, int height, int channels){
    int gray_channels = channels == 4 ? 2 : 1;
    size_t gray_img_size = width * height * gray_channels;

    unsigned char *gray_img = (unsigned char *)malloc(gray_img_size);
    if(gray_img == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int index = j + width * i;
            unsigned char* pixel = img + index * channels;
            unsigned char* gray_pixel = gray_img + index * gray_channels;

            *gray_pixel = (uint8_t)((pixel[0] + pixel[1] + pixel[2])/3.0);
            if(channels == 4) {
                gray_pixel[1] = pixel[3];
            }
        }
    }

    cout << "Generated Gray image" << endl; 

    stbi_write_png("../images/gray.png", width, height, gray_channels, gray_img, width * gray_channels);

    cout << "Wrote Gray image" << endl; 

    free(gray_img);
}

// stb use from: https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
int main(void) {

    cout << "Starting Sequential Version..." << endl;

    

    int width, height, channels;
    unsigned char *img = stbi_load("../images/beach.png", &width, &height, &channels, 0);
    unsigned char *og_img = stbi_load("../images/beach.png", &width, &height, &channels, 0);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    double startTime = CycleTimer::currentSeconds();

    cout << "Read image, width: " <<  width << ", height: " << height << ", channels: "<<channels << endl; 
    color_dither(img, width, height, channels, 4);

    double endTime = CycleTimer::currentSeconds();

    cout << "Time Taken: " << endTime - startTime << " seconds" << endl;

    //gray(img, width, height, channels);

    cout << "Starting Parallel Version..." << endl;

    printCudaInfo();
    pfsCuda(width, height, channels, og_img);


    stbi_image_free(img);
    stbi_image_free(og_img);

    
    return 0;
}