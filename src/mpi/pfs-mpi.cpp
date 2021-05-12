#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <math.h>
#include "mpi.h"


#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image/stb_image_write.h"

#define FACTOR 4

using namespace std;

// Tag translations:
// 0 = send width
// 1 = send height
// 2 = send original image
// 3 = send processed image
// 4 = computer thread done message


double computeThread(int processCount, int processID){
  int width;
  int height;
  int channels = 3;
  int numRowsPerProc;
  int computeProcID;
  int startInd;
  int endInd;

  MPI_Status status;
  int receivedInt;
  unsigned char *image;
  int flag = 0;
  bool wait = true;

  // Get Width and Height from Reader thread
  while(wait) {
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
    flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;
    if(tag <= 1){
      MPI_Get_count(&status, MPI_INT, &count);
      MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
      if(tag == 0){
        // Receive width
        width = receivedInt;
      } else if (tag == 1){
        // Receive height, and set up information to operate on individual row chunk
        height = receivedInt;

        int totalComputeProcs = processCount - 2;
        numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
        computeProcID = processID - 2;
        startInd = width*channels*numRowsPerProc*computeProcID;
        endInd = width*channels*numRowsPerProc*(computeProcID+1);
      }
    } else {
      // Receive orginal image's block of rows
      image = (unsigned char *) malloc(width * numRowsPerProc * channels * sizeof(unsigned char));
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
      MPI_Recv(image, count, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status);

      wait = false;
    }
  }

  // Start to dither individual block
  for(int y = 0; y < numRowsPerProc; y++){
    for(int x = 0; x < width; x++){

      unsigned char* pixel = image + (x + width * y) * channels;
      
      unsigned char* pixel_right = NULL;
      unsigned char* pixel_bottom_left = NULL;
      unsigned char* pixel_bottom = NULL;
      unsigned char* pixel_bottom_right = NULL;

      if(x+1 < width){
        pixel_right = image + ((x+1) + width * y) * channels;
      }
      
      if(x-1 >= 0 && y+1 < numRowsPerProc){
        pixel_bottom_left = image + ((x-1) + width * (y+1)) * channels;
      }
      
      if(y+1 < numRowsPerProc){
        pixel_bottom = image + (x + width * (y+1)) * channels;
      }
      
      if(x+1 < width && y+1 < numRowsPerProc){
        pixel_bottom_right = image + ((x+1) + width * (y+1)) * channels;
      }

      int oldR = static_cast<int>(pixel[0]);
      int oldG = static_cast<int>(pixel[1]);
      int oldB = static_cast<int>(pixel[2]);

      int newR = round(FACTOR * oldR / 255.0) * (255/FACTOR);
      int newG = round(FACTOR * oldG / 255.0) * (255/FACTOR);
      int newB = round(FACTOR * oldB / 255.0) * (255/FACTOR);

      int qErrorR = oldR - newR;
      int qErrorG = oldG - newG;
      int qErrorB = oldB - newB;

      pixel[0] = newR;
      pixel[1] = newG;
      pixel[2] = newB;

      if(pixel_right != NULL){
        pixel_right[0] = (uint8_t)(pixel_right[0] + (qErrorR * (7.0 / 16.0)));
        pixel_right[1] = (uint8_t)(pixel_right[1] + (qErrorG * (7.0 / 16.0)));
        pixel_right[2] = (uint8_t)(pixel_right[2] + (qErrorB * (7.0 / 16.0)));
      }
      
      if(pixel_bottom_left != NULL){
        pixel_bottom_left[0] = (uint8_t)(pixel_bottom_left[0] + (qErrorR * (3.0 / 16.0)));
        pixel_bottom_left[1] = (uint8_t)(pixel_bottom_left[1] + (qErrorG * (3.0 / 16.0)));
        pixel_bottom_left[2] = (uint8_t)(pixel_bottom_left[2] + (qErrorB * (3.0 / 16.0)));
      }

      if(pixel_bottom != NULL){
        pixel_bottom[0] = (uint8_t)(pixel_bottom[0] + (qErrorR * (5.0 / 16.0)));
        pixel_bottom[1] = (uint8_t)(pixel_bottom[1] + (qErrorG * (5.0 / 16.0)));
        pixel_bottom[2] = (uint8_t)(pixel_bottom[2] + (qErrorB * (5.0 / 16.0)));
      }

      if(pixel_bottom_right != NULL){
        pixel_bottom_right[0] = (uint8_t)(pixel_bottom_right[0] + (qErrorR * (1.0 / 16.0)));
        pixel_bottom_right[1] = (uint8_t)(pixel_bottom_right[1] + (qErrorG * (1.0 / 16.0)));
        pixel_bottom_right[2] = (uint8_t)(pixel_bottom_right[2] + (qErrorB * (1.0 / 16.0))); 
      }

    }
  }

  // Send completion signal to Read thread
  MPI_Request request1;
  int temp = 1;
  MPI_Isend(&temp, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &request1);
  MPI_Request_free(&request1);

  // Send dithered block to Write thread
  MPI_Request request2;
  MPI_Isend(image, width * numRowsPerProc * channels, MPI_UNSIGNED_CHAR, 1, 3, MPI_COMM_WORLD, &request2);
  MPI_Request_free(&request2);

  return 0.0;
}

double writeThread(int processCount){
  int width;
  int height;
  int channels = 3;
  int numRowsPerProc;
  int totalElems;

  MPI_Status status;
  int receivedInt;
  int flag = 0;
  bool wait = true;
  unsigned char *finalImage;

  // Get Width and Height from Reader thread
  while(wait) {
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
    flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    if(tag == 0){
      // Receive width
      width = receivedInt;
    } else if (tag == 1){
      // Receive height and setup the final image
      height = receivedInt;
      totalElems = width * height * channels;
      finalImage = (unsigned char *) malloc(totalElems * sizeof(unsigned char));
      int totalComputeProcs = processCount - 2;
      numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
      wait = false;
    }
  }

  wait = true;
  int finishedProcesses = 0;
  // Get parts of computed array
  while(wait){
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
    flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;

    if(tag == 3){
      // Receive a row chunk from a compute processor
      finishedProcesses += 1;
      unsigned char *imageFromProc = (unsigned char *) malloc(width * numRowsPerProc * channels * sizeof(unsigned char));

      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
      MPI_Recv(imageFromProc, count, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status);

      int computeProcID = source - 2;
      int startInd = width*channels*numRowsPerProc*computeProcID;
      int endInd = width*channels*numRowsPerProc*(computeProcID+1);

      // Copy over to the final image in the correct block spot
      for(int i = startInd; i < endInd && i < totalElems; i++ ){
        finalImage[i] = imageFromProc[i-startInd];
      }

      free(imageFromProc);

      if(finishedProcesses == processCount - 2){
        // Break out if all compute procs responded
        wait = false;
        break;
      }
    }
  }

  // Image should be finished now, write out
  stbi_write_png("../../images/dither-mpi.png", width, height, channels, finalImage, width * channels);

  free(finalImage);
  return 0.0;

}

double readThread(int processCount){
  int totalComputeProcs = processCount - 2;  
  
  int width;
  int height;
  int channels;

  // Load image
  unsigned char *img = stbi_load("../../images/roadster.png", &width, &height, &channels, 0);

  int numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;

  for(int proc = 1; proc < processCount; proc++){
    // Send width and height of image
    MPI_Request request1;
    MPI_Isend(&width, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, &request1);
    MPI_Request_free(&request1);

    MPI_Request request2;
    MPI_Isend(&height, 1, MPI_INT, proc, 1, MPI_COMM_WORLD, &request2);
    MPI_Request_free(&request2);
    
    // If sending to a compute proc, also send individual chunk of image
    if(proc > 1){
      int computeProcID = proc - 2;
      int startInd = width*numRowsPerProc*channels*computeProcID;

      MPI_Request request3;
      MPI_Isend(img + startInd, width * numRowsPerProc * channels, MPI_UNSIGNED_CHAR, proc, 2, MPI_COMM_WORLD, &request3);
      MPI_Request_free(&request3);
    }
  }

  MPI_Status status;
  int receivedInt;
  int flag = 0;
  bool wait = true;
  int finishedProcesses = 0;

  // Wait for compute processes to signal they have finished
  while(wait) {
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
    flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    if(tag == 4){
      finishedProcesses += 1;
      if(finishedProcesses == processCount - 2){
        // If all compute procs have finished, return end time
        return MPI_Wtime();
      }
    }
  }
  return 0.0;
}

int main() {

  int processID, processCount;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank( MPI_COMM_WORLD, &processID );
  MPI_Comm_size( MPI_COMM_WORLD, &processCount );


  double t1 = MPI_Wtime();
  double t2;
  if(processID == 0){
    t2 = readThread(processCount);
    printf("============ Time ============\n");
    printf("Time: %f s\n", t2-t1);
  } else if(processID == 1){
    writeThread(processCount);
  } else {
    computeThread(processCount, processID);
  }
  
  MPI_Finalize();

    return 0;
}