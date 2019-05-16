/* Game of Life - CUDA implementation.
 * Using multiple cells per CUDA thread and shared memory.
 *
 * Copyright (C) 2019 Alexandros I. Metsai
 * alexmetsai@gmail.com
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define THRESHOLD 0.4
#define CELLS_PER_THREAD 2
#define THREADS_PER_BLOCK (500/CELLS_PER_THREAD)


struct timeval startwtime, endwtime;
double seq_time;


__global__ void game_c (int *newer, int *old, int N)
{
	
  int lsize=THREADS_PER_BLOCK*CELLS_PER_THREAD;
  
  __shared__ int top[THREADS_PER_BLOCK*CELLS_PER_THREAD+2];		//Extended Tables
  __shared__ int mid[THREADS_PER_BLOCK*CELLS_PER_THREAD+2];
  __shared__ int bot[THREADS_PER_BLOCK*CELLS_PER_THREAD+2];
  
  int index = blockIdx.x*blockDim.x*CELLS_PER_THREAD + threadIdx.x*CELLS_PER_THREAD;
  int count;
  int sum=0;
  int i=(int) index/N;
  int j= index%N;
  int lindex = threadIdx.x*CELLS_PER_THREAD +1; //Local Index
  
  for(count=0; count<CELLS_PER_THREAD;count++){
	  
	  
	  if(i==0){
		  
		  //top
		  
		  if(j==0){
			  
			  top[0]= old[N*N-1];
			  mid[0]= old[N-1];
			  bot[0]= old[2*N -1];
			  
			  top[1]= old[N*(N-1)];
			  mid[1]= old[0];
			  bot[1]= old[N];
			  
		  }else if(j==(N-1)){
			  
			  top[lindex+1]= old[N*(N-1)];
			  mid[lindex+1]= old[0];
			  bot[lindex+1]= old[N];
			  
			  top[lindex]= old[N*N -1];
			  mid[lindex]= old[N-1];
			  bot[lindex]= old[2*N -1];
			  
		  }else{
			  if(lindex==1){
				  
				  top[lindex-1]= old[N*(N-1) +(j-1)];
				  mid[lindex-1]= old[j-1];
				  bot[lindex-1]= old[N+(j-1)];
				  
				  top[lindex]= old[N*(N-1) +j];
				  mid[lindex]= old[j];
				  bot[lindex]= old[N+j];
				  
			  }else if(lindex==(lsize)){
				  
				  top[lindex+1]= old[N*(N-1) +(j+1)];
				  mid[lindex+1]= old[j+1];
				  bot[lindex+1]= old[N+ (j+1)];
				  
				  top[lindex]= old[N*(N-1) +j];
				  mid[lindex]= old[j];
				  bot[lindex]= old[N +j];
				  
			  }else{
				  
				  top[lindex]= old[N*(N-1) +j];
				  mid[lindex]= old[j];
				  bot[lindex]= old[N+j];
				  
			  }
		  }
	  }else if(i==(N-1)){
		  
		  //bottom
		  
		  if(j==0){
			  
			  top[0]= old[N*(N-1) -1];
			  mid[0]= old[N*N -1];
			  bot[0]= old[N-1];
			  
			  top[1]= old[N*(N-2)];
			  mid[1]= old[N*(N-1)];
			  bot[1]= old[0];
			  
		  }else if(j==(N-1)){
			  
			  top[lindex+1]= old[N*(N-2)];
			  mid[lindex+1]= old[N*(N-1)];
			  bot[lindex+1]= old[0];
			  
			  top[lindex]= old[N*(N-1) -1];
			  mid[lindex]= old[N*N -1];
			  bot[lindex]= old[N-1];
			  
		  }else{
			  // !!
			  if(lindex==1){
			  
				top[lindex-1]= old[(i-1)*N +(j-1)];
				mid[lindex-1]= old[i*N +(j-1)];
				bot[lindex-1]= old[j-1];
			  
				top[lindex]= old[(i-1)*N +j];
				mid[lindex]= old[i*N +j];
				bot[lindex]= old[j];
			  
			  }else if(lindex==(lsize)){
			  
			   top[lindex+1]= old[(i-1)*N +(j+1)];
			   mid[lindex+1]= old[i*N +(j+1)];
			   bot[lindex+1]= old[(j+1)];
			  
			   top[lindex]= old[(i-1)*N +j];
			   mid[lindex]= old[i*N +j];
			   bot[lindex]= old[j];
			   
		    }else{
		  
				top[lindex]= old[(i-1)*N +j];
				mid[lindex]= old[i*N +j];
				bot[lindex]= old[j];
			}
			  
		 }
	  }else if(j==0){
		  
		  //left
		  
		  top[0]= old[(i-1)*N +(N-1)];
		  mid[0]= old[i*N +(N-1)];
		  bot[0]= old[(i+1)*N +(N-1)];
		  
		  top[1]= old[(i-1)*N];
		  mid[1]= old[i*N];
		  bot[1]= old[(i+1)*N];
		  
	  }else if(j==(N-1)){
		  
		  //right
		  
		  top[lindex+1]= old[(i-1)*N];
		  mid[lindex+1]= old[i*N];
		  bot[lindex+1]= old[(i+1)*N];
		  
		  top[lindex]= old[(i-1)*N +j];
		  mid[lindex]= old[i*N +j];
		  bot[lindex]= old[(i+1)*N +j];
		  
	  }else{
		  
		  //general case
		  
		  if(lindex==1){
			  
			  top[lindex-1]= old[(i-1)*N +(j-1)];
			  mid[lindex-1]= old[i*N +(j-1)];
			  bot[lindex-1]= old[(i+1)*N +(j-1)];
			  
			  top[lindex]= old[(i-1)*N +j];
			  mid[lindex]= old[i*N +j];
			  bot[lindex]= old[(i+1)*N +j];
			  
		  }else if(lindex==(lsize)){
			  
			  top[lindex+1]= old[(i-1)*N +(j+1)];
			  mid[lindex+1]= old[i*N +(j+1)];
			  bot[lindex+1]= old[(i+1)*N +(j+1)];
			  
			  top[lindex]= old[(i-1)*N +j];
			  mid[lindex]= old[i*N +j];
			  bot[lindex]= old[(i+1)*N +j];
		  }else{
		  
			top[lindex]= old[(i-1)*N +j];
			mid[lindex]= old[i*N +j];
			bot[lindex]= old[(i+1)*N +j];
			
		  }
	  }
	  
	  lindex++;
	  j++;
  }
  
  //Restore values
  j=index%N;
  lindex=threadIdx.x*CELLS_PER_THREAD +1;
  
  __syncthreads();
  
  for(count=0; count<CELLS_PER_THREAD; count++){
	  
	  sum= top[lindex-1] +top[lindex]+top[lindex+1]
	  +mid[lindex-1] +mid[lindex+1]
	  +bot[lindex-1] +bot[lindex] +bot[lindex+1];
	  
	  switch(sum){
		  
		  case 3:
		  newer[i*N + j] = 1;
		  break;
		  
		  case 2:
		  newer[i*N + j] = old[i*N + j];
		  break;
		  
		  default:
		  newer[i*N + j] = 0;
	  }
	  
	  lindex++;
	  j++;
  }

}


void read_from_file(int *X, char *filename, int N);
void save_table(int *X, int N);


int main(){
	

	int *table;
	int* newer;
	int* old;
	int *temp;
	
	int blocks, t, N, count;

	
	printf("Set the number of generations\n");
	scanf("%d", &t);
	printf("Set N (table size = NxN)\n");
	scanf("%d", &N);
	int size=N*N*sizeof(int);
	
	/*
	 Insert table here
	 */
	
	
	char filename[20];
	sprintf(filename, "table%dx%d.bin", N, N);
	printf("Reading %dx%d table from file %s\n", N, N, filename);
	table = (int *)malloc(N*N*sizeof(int));
	read_from_file(table, filename, N);  	
	

	printf("This is kernel c\n");
	
	printf("The game will be played for %d generations N=%d\n", t, N);
	
	//!!!Start Timer!!!
	gettimeofday (&startwtime, NULL);
	
	//Allocate space of new and old in device
	cudaMalloc(&newer, size);
	cudaMalloc(&old, size);
	
	//copy table
	cudaMemcpy(old, table, size, cudaMemcpyHostToDevice);
	
	blocks=(N*N)/(THREADS_PER_BLOCK*CELLS_PER_THREAD);
	
	//Play game for t generations
	for(count=0;count<t;count++){
		
		game_c<<<blocks, THREADS_PER_BLOCK>>>(newer, old, N);
		cudaThreadSynchronize();
		
		//swap pointers
		temp=old;
		old=newer;
		newer=temp;
	}
	
	//copy back table
	cudaMemcpy(table, old, size, cudaMemcpyDeviceToHost);
	
	//!!!End Timer!!!
	gettimeofday (&endwtime, NULL);
	  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	  printf("Cuda clock time = %f\n", seq_time);
	
	save_table(table, N);
	
	cudaFree(newer);
	cudaFree(old);
	
	free(table);
	
	return(0);
}


void read_from_file(int *X, char *filename, int N){

  FILE *fp = fopen(filename, "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  printf("elements: %d\n", size);

  fclose(fp);

}

void save_table(int *X, int N){

  FILE *fp;

  char filename[20];

  sprintf(filename, "cuda_c_table%dx%d.bin", N, N);

  printf("Saving table in file %s\n", filename);

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);

}
