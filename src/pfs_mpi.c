#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <math.h>
#include "mpi.h"


#define SYSEXPECT(expr) do { if(!(expr)) { perror(__func__); exit(1); } } while(0)
#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);


void findPaths_seq(int currCityID, int *path, int *visited, int visitedPos, int dist, int *minDist, int*minPath){
  path[visitedPos] = currCityID;
  visited[currCityID] = 1;

  
  if(dist > *minDist){
    visited[currCityID] = 0;
    return;
  }
  
  if(visitedPos == NCITIES-1){
    if(dist < *minDist){
      *minDist = dist;
      for(int i = 0; i < NCITIES; i++){
        minPath[i] = path[i];
      }
      
    }
    visited[currCityID] = 0;
    return;
  }

  for (int i = 0; i < NCITIES; i++) {
    if(visited[i] == 0){
      findPaths_seq(i, path, visited, visitedPos+1, dist+get_dist(currCityID, i), minDist, minPath);
    }
  }

  visited[currCityID] = 0;
  return;
}

int *wsp_start_seq(int *minDist) {
  int *minPath = (int *) malloc((NCITIES+1) * sizeof(int));
  for(int start_id = 0; start_id < NCITIES; start_id++){
    for(int child = 0; child < NCITIES; child++){
        if(child != start_id){
            int path[NCITIES + 1];
            path[0] = start_id;
            int visited[NCITIES];
            for(int i = 0; i < NCITIES; i++){
                visited[i] = 0;
            }
            visited[start_id] = 1;

            findPaths_seq(child, path, visited, 1, get_dist(start_id, child), minDist, minPath);
        }
        
    }
  }
  return minPath;
}



void findPaths(int currCityID, int *path, int *visited, int visitedPos, int dist, grid_t *g){
  path[visitedPos] = currCityID;
  visited[currCityID] = 1;
  
  if(dist > g->minDist){
    visited[currCityID] = 0;
    return;
  }
  
  if(visitedPos == NCITIES-1){
    if(dist < g->minDist){
      g->minDist = dist;
      
      path[NCITIES] = dist;
      
      MPI_Request path_request;
      MPI_Isend(path, NCITIES+1, MPI_INT, 0, 1, MPI_COMM_WORLD, &path_request);
      // Don't need to wait, just free immediately
      MPI_Request_free(&path_request);
    }
    
    visited[currCityID] = 0;
    return;
  }

  for (int i = 0; i < NCITIES; i++) {
    if(visited[i] == 0){
      findPaths(i, path, visited, visitedPos+1, dist+get_dist(currCityID, i), g);
    }
  }

  visited[currCityID] = 0;
  return;
}

// may need to remove some of this
grid_t *new_grid(int processID, int processCount, int minDist){
  grid_t *g = (grid_t *) malloc(sizeof(grid_t));
  g->processID = processID;
  g->processCount = processCount;
  g->totalElems = pow(NCITIES, 2);
  g->elemCount = (g->totalElems + (processCount-1) - 1)/(processCount-1);
  g->elemStart = g->elemCount * (processID-1);
  g->gridDim = NCITIES;
  g->minDist = minDist;
  return g;
}

void free_grid(grid_t *g){
  free(g);
}

void wsp_start(int processID, int processCount, int minDist) {
  grid_t *g = new_grid(processID, processCount, minDist);

  for(int elem = g->elemStart; elem < g->totalElems && elem < g->elemStart + g->elemCount; elem++){
    int start_id = elem/g->gridDim;
    int child = elem % g->gridDim;

    if(child != start_id){
        int path[NCITIES + 1];
        path[0] = start_id;
        int visited[NCITIES];
        for(int i = 0; i < NCITIES; i++){
            visited[i] = 0;
        }
        visited[start_id] = 1;

        findPaths(child, path, visited, 1, get_dist(start_id, child), g);
      }
  }

  MPI_Request finish_request;
  int temp = 1;
  MPI_Isend(&temp, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &finish_request);
  // Don't need to wait, just free immediately
  MPI_Request_free(&finish_request);

  free_grid(g);
  return;
}

int *track_paths(int processCount){
  int *globalMinPath = (int *) malloc((NCITIES+1) * sizeof(int));
  int globalMinDist = INT_MAX;
  int finishedProcesses = 0;
  int flag = 0;
  int receivedPath[NCITIES+1];
  MPI_Status status;
  while(1) {
    while(!flag){
       MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
     }
     flag = 0;
    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int count;
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Recv(&receivedPath, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    if(tag == 0){
      finishedProcesses += 1;
      if(finishedProcesses == processCount - 1){
        return globalMinPath;
      }
    } else if (tag == 1){
      if(receivedPath[NCITIES] < globalMinDist){
        //Save Dist
        globalMinDist = receivedPath[NCITIES];
        // Copy over path
        for(int i = 0; i < NCITIES+1; i++){
          globalMinPath[i] = receivedPath[i];
        }
        
      }
    }
  }
  return NULL;
}

void getBound(int currCityID, int *path, int *visited, int visitedPos, int dist, int *minDist){
  
  path[visitedPos] = currCityID;
  visited[currCityID] = 1;

  if(visitedPos == NCITIES-1){
    *minDist = dist;
    return;
  }

  int minDiff = INT_MAX;
  int minChild = -1;
  for (int i = 0; i < NCITIES; i++) {
    int diff = get_dist(currCityID, i);
    if(visited[i] == 0 && diff < minDiff){
      minDiff = diff;
      minChild = i;
    }
  }

  getBound(minChild, path, visited, visitedPos+1, dist+minDiff, minDist);

  return;
}

int main(int argc, char **argv) {
  //printf("%d", argc);
  if(argc < 2) error_exit("Expecting one arguments: [file name]\n");
  // NCORES = atoi(argv[2]);
  // if(NCORES < 1) error_exit("Illegal core count: %d\n", NCORES);
  char *filename = argv[1];
  FILE *fp = fopen(filename, "r");
  if(fp == NULL) error_exit("Failed to open input file \"%s\"\n", filename);
  int scan_ret;
  scan_ret = fscanf(fp, "%d", &NCITIES);
  if(scan_ret != 1) error_exit("Failed to read city count\n");
  if(NCITIES < 2) {
    error_exit("Illegal city count: %d\n", NCITIES);
  } 
  // Allocate memory and read the matrix
  DIST = (int*)calloc(NCITIES * NCITIES, sizeof(int));
  SYSEXPECT(DIST != NULL);
  for(int i = 1;i < NCITIES;i++) {
    for(int j = 0;j < i;j++) {
      int t;
      scan_ret = fscanf(fp, "%d", &t);
      if(scan_ret != 1) error_exit("Failed to read dist(%d, %d)\n", i, j);
      set_dist(i, j, t);
      set_dist(j, i, t);
    }
  }
  fclose(fp);
  bestPath = (path_t*)malloc(sizeof(path_t));
  bestPath->cost = 0;
  bestPath->path = (city_t*)calloc(NCITIES, sizeof(city_t));

  int processID, processCount;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank( MPI_COMM_WORLD, &processID );
  MPI_Comm_size( MPI_COMM_WORLD, &processCount );
  double t1 = MPI_Wtime();

  int minDist = INT_MAX;

  int *minPath =(int *)malloc((NCITIES+1)*sizeof(int));
  if(processCount == 1){
    minPath = wsp_start_seq(&minDist);
  } else {
    if(processID == 0){
      minPath = track_paths(processCount);
    } else{
      wsp_start(processID, processCount, minDist);
    }
  }
  
  MPI_Finalize();

  double t2 = MPI_Wtime();

  if(processID == 0) {

    if(processCount == 1){
      for(int i = 0; i < NCITIES; i++){
        bestPath->path[i] = minPath[i];
      }
      bestPath->cost = minDist;
    } else {
      for(int i = 0; i < NCITIES; i++){
        bestPath->path[i] = minPath[i];
      }
      bestPath->cost = minPath[NCITIES];
    }
    
    printf("============ Time ============\n");
    printf("Time: %f s\n", t2-t1);
    wsp_print_result();
    free(minPath);
  }
  return 0;
}