#include <stdio.h>

__global__ void cuda_hello() { 
   printf("Hello from Block %d, Thread %d!\n", blockIdx.x, threadIdx.x);
}

int main() { 
   cuda_hello<<<3,3>>>(); 
   cudaDeviceSynchronize();
   return 0; 
}
