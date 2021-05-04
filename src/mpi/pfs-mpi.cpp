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

double computeThread(int processID){
  int width;
  int height;
  int channels = 3;

  MPI_Status status;
  int receivedInt;
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
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    if(tag == 0){
      width = receivedInt;
      //cout << "Proc " << processID << ": width=" << width << endl;
    } else if (tag == 1){
      height = receivedInt;
      //cout << "Proc " << processID <<": height=" << height << endl;
      wait = false;
    }
  }

  return 0.0;
}

double writeThread(int processCount){
  int width;
  int height;
  int channels = 3;

  MPI_Status status;
  int receivedInt;
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
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Recv(&receivedInt, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    if(tag == 0){
      width = receivedInt;
      //cout << "Proc 1: width=" << width << endl;
    } else if (tag == 1){
      height = receivedInt;
      //cout << "Proc 1: height=" << height << endl;
      wait = false;
    }
  }

  return 0.0;

}

// Tag translations:
// 0 = send width
// 1 = send height


double readThread(int processCount){
  int totalComputeProcs = processCount - 2;
  int width = 5;
  int height = 12;
  int channels = 3;

  unsigned char *img = (unsigned char *) calloc(width * height * channels, sizeof(unsigned char));
  
  for(int proc = 1; proc < processCount; proc++){
    MPI_Request request1;
    MPI_Isend(&width, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, &request1);
    MPI_Request_free(&request1);
    MPI_Request request2;
    MPI_Isend(&height, 1, MPI_INT, proc, 1, MPI_COMM_WORLD, &request2);
    MPI_Request_free(&request2);
    cout << "Proc 0: Sent width and height to proc " << proc << endl;
  }
  
  int numRowsPerProc = (height + (totalComputeProcs - 1)) / totalComputeProcs;
  cout << "Proc 0: numRowsPerProc=" << numRowsPerProc << endl;
  unsigned char *img_blocks = (unsigned char *) malloc(width * height * channels * sizeof(unsigned char));

  for(int proc = 0; proc < totalComputeProcs; proc++){
    cout << "Proc 0: For proc " << proc << " -> Starting from index " << width*channels*numRowsPerProc*proc << " to " << width*channels*numRowsPerProc*(proc+1 << endl;
    int block_ind = 0;
    for(int i = width*channels*numRowsPerProc*proc; i < width*channels*numRowsPerProc*(proc+1) and i < height; i++ ){
      cout << "Proc 0: index " << i << endl;
      img_blocks[proc][block_ind] = img[i];
      block_ind++;
    }
    cout << "Proc 0: Sent to proc " << proc << endl;
  }

  return MPI_Wtime();
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
    computeThread(processID);
  }
  
  MPI_Finalize();

  if(processID == 0) {
    printf("============ Time ============\n");
    printf("Time: %f s\n", t2-t1);
  }
//   return 0;
    return 0;
}