#include <stdio.h>

__global__ void cuda_hello() { 
   printf("Hello World!\n"); 
}

int main() { 
   cuda_hello<<<6,1>>>();
   cudaDeviceSynchronize();
   return 0; 
}
