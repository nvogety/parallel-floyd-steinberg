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

using namespace std;

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
  unsigned char *receivedImage;
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
    //cout << "Proc " << processID << ": Received msg with tag " << tag << endl;
    if(tag <= 1){
      MPI_Get_count(&status, MPI_INT, &count);
      MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
      if(tag == 0){
        width = receivedInt;
        //cout << "Proc " << processID << ": width=" << width << endl;
      } else if (tag == 1){
        height = receivedInt;
        //cout << "Proc " << processID <<": height=" << height << endl;
        int totalComputeProcs = processCount - 2;
        numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
        computeProcID = processID - 2;
        startInd = width*channels*numRowsPerProc*computeProcID;
        endInd = width*channels*numRowsPerProc*(computeProcID+1);
      }
    } else {
      //cout << "Proc " << processID <<": In image receive" << endl;
      //cout << "Proc " << processID << ": width=" << width <<  ", height=" << height << ", numRows=" << numRowsPerProc <<  ", startInd="<< startInd << ", endInd=" << endInd << endl;
      image = (unsigned char *) malloc(width * height * channels * sizeof(unsigned char));
      //cout << "Proc " << processID <<": Mallocd" << endl;
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
      //cout << "Proc " << processID <<": Count of elems=" << count << endl;
      MPI_Recv(image, count, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status);
      //cout << "Proc " << processID <<": Received image"  << endl;
      wait = false;
      // cout << "Proc " << processID <<": endInd-startInd=" << endInd - startInd << ", total elems mallocd=" << width * numRowsPerProc * channels << endl;
      // int totalElems = width * height * channels;
      // for(int i = startInd; i < endInd and i < totalElems; i++ ){
      //   cout << "Proc " << processID <<": Copying elem " << i << endl;
      //   image[startInd] = receivedImage[i-startInd];
      // }
    }
  }
  int totalElems = width * height * channels;
  for(int i = startInd; i < endInd && i < totalElems; i++ ){
      int val = static_cast<int>(image[i]);
      image[i] = (uint8_t)(1);
      image[i] = (uint8_t)(2);
      image[i] = (uint8_t)(3);

      image[i] = (uint8_t)(4);
      image[i] = (uint8_t)(5);
      image[i] = (uint8_t)(6);

      image[i] = (uint8_t)(7);
      image[i] = (uint8_t)(8);
      image[i] = (uint8_t)(9);

      image[i] = (uint8_t)(10);
      image[i] = (uint8_t)(11);
      image[i] = (uint8_t)(12);

      image[i] = (uint8_t)(13);
      image[i] = (uint8_t)(14);
      image[i] = (uint8_t)(15);

      image[i] = (uint8_t)(val+processID);
      
  }
  cout << "Proc " << processID <<": Finished Copying elems"<< endl;
  

  MPI_Request request1;
  int temp = 1;
  MPI_Isend(&temp, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &request1);
  MPI_Request_free(&request1);
  cout << "Proc " << processID <<": sent finished message to proc 0 "<< endl;

  MPI_Request request2;
  cout << "Proc " << processID <<": Before sent computed elems to proc 1 "<< endl;
  MPI_Isend(image, width * height * channels, MPI_UNSIGNED_CHAR, 1, 3, MPI_COMM_WORLD, &request2);
  MPI_Request_free(&request2);
  cout << "Proc " << processID <<": sent computed elems to proc 1 "<< endl;

  free(image);
  // cout << "Proc " << processID <<": Freed image "<< endl;

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
      width = receivedInt;
      cout << "Proc 1: width=" << width << endl;
    } else if (tag == 1){
      height = receivedInt;
      cout << "Proc 1: height=" << height << endl;
      totalElems = width * height * channels;
      cout << "Proc 1: totalElems=" << totalElems << endl;
      finalImage = (unsigned char *) malloc(totalElems * sizeof(unsigned char));
      cout << "Proc 1: finalImage mallocd" << endl;
      int totalComputeProcs = processCount - 2;
      numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
      cout << "Proc 1: rowsRowsPerProc=" << numRowsPerProc << endl;
      wait = false;
    }
  }

  wait = true;
  int finishedProcesses = 0;
  cout << "Proc 1: Listening for computations" << endl;
  // Get parts of computed array
  while(wait){
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
    flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;
    cout << "Proc 1: Recieved computation from proc " << source << " with tag " << tag << endl;
    if(tag == 3){
      finishedProcesses += 1;
      unsigned char *imageFromProc = (unsigned char *) malloc(width * height * channels * sizeof(unsigned char));
      cout << "Proc 1: Malloc'd imageFromProc " << endl;
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
      MPI_Recv(imageFromProc, count, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status);
      cout << "Proc 1: Received imageFromProc " << endl;
      int computeProcID = source - 2;
      int startInd = width*channels*numRowsPerProc*computeProcID;
      int endInd = width*channels*numRowsPerProc*(computeProcID+1);
      cout << "Proc 1: computeProcID = " << computeProcID << ",startInd = " << startInd << ", endInd = " << endInd << endl;
      for(int i = startInd; i < endInd && i < totalElems; i++ ){
        finalImage[i] = imageFromProc[i];
      }
      cout << "Proc 1: Copied over imageFromProc to finalImage " << endl;
      free(imageFromProc);
      cout << "Proc 1: Freed imageFromProc " << endl;
      cout << "Proc 1: finishedProcesses=" << finishedProcesses << endl;
      if(finishedProcesses == processCount - 2){
        cout << "Proc 1: All processes finished, breaking" << endl;
        wait = false;
        break;
      }
    }
  }

  cout << "Proc 1: Reading finalImage now" << endl;
  for(int i = 0; i < totalElems; i++ ){
    if (i == 0 || finalImage[i] != finalImage[i-1]){
      int val = static_cast<int>(finalImage[i]);
      cout << "finalImage[" << i <<  "] = " << val-2 << endl;
    }
  }
  
  cout << "Proc 1: Returning" << endl;
  free(finalImage);
  return 0.0;

}

// Tag translations:
// 0 = send width
// 1 = send height
// 2 = send original image
// 3 = send processed image
// 4 = computer thread done message


double readThread(int processCount){
  //int totalComputeProcs = processCount - 2;
  int width = 200;
  int height = 200;
  int channels = 3;

  unsigned char *img = (unsigned char *) calloc(width * height * channels, sizeof(unsigned char));
  
  for(int proc = 1; proc < processCount; proc++){
    MPI_Request request1;
    MPI_Isend(&width, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, &request1);
    MPI_Request_free(&request1);
    MPI_Request request2;
    MPI_Isend(&height, 1, MPI_INT, proc, 1, MPI_COMM_WORLD, &request2);
    MPI_Request_free(&request2);
    
    if(proc > 1){
      MPI_Request request3;
      MPI_Isend(img, width * height * channels, MPI_UNSIGNED_CHAR, proc, 2, MPI_COMM_WORLD, &request3);
      MPI_Request_free(&request3);
    }
    
    cout << "Proc 0: Sent width, height, and image to proc " << proc << endl;
  }

  MPI_Status status;
  int receivedInt;
  int flag = 0;
  bool wait = true;
  int finishedProcesses = 0;

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
      // cout << "Proc 0: proc "<<source << " finished, finProcs=" << finishedProcesses << endl;
      if(finishedProcesses == processCount - 2){
        cout << "Proc 0: all processes finished, returning" << endl;
        return MPI_Wtime();
      }
    }
  }

  // int numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
  // cout << "Proc 0: numRowsPerProc=" << numRowsPerProc << endl;
  // unsigned char *img_blocks = (unsigned char *) malloc(width * height * channels * sizeof(unsigned char));

  // for(int proc = 0; proc < totalComputeProcs; proc++){
  //   cout << "Proc 0: For proc " << proc << " -> Starting from index " << width*channels*numRowsPerProc*proc << " to " << width*channels*numRowsPerProc*(proc+1 << endl;
  //   int block_ind = 0;
  //   for(int i = width*channels*numRowsPerProc*proc; i < width*channels*numRowsPerProc*(proc+1) and i < height; i++ ){
  //     cout << "Proc 0: index " << i << endl;
  //     img_blocks[proc][block_ind] = img[i];
  //     block_ind++;
  //   }
  //   cout << "Proc 0: Sent to proc " << proc << endl;
  // }

  return 0.0;
}

int main(int argc, char **argv) {

  // if(argc < 2){
  //   cout << "Enter a filename" << endl;
  //   return -1;
  // }

  // char *filename = argv[1];

  int processID, processCount;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank( MPI_COMM_WORLD, &processID );
  MPI_Comm_size( MPI_COMM_WORLD, &processCount );

  // cout << "Hi from process" << processID << endl;

  double t1 = MPI_Wtime();
  double t2;
  int width, height, channels;
  if(processID == 0){
    t2 = readThread(processCount);
  } else if(processID == 1){
    writeThread(processCount);
  } else {
    computeThread(processCount, processID);
  }
  
  MPI_Finalize();

  if(processID == 0) {
    printf("============ Time ============\n");
    printf("Time: %f s\n", t2-t1);
  }
//   return 0;
    return 0;
}