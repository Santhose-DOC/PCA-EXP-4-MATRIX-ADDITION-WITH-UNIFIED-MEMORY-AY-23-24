# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME : SANTHOSE AROCKIARAJ J</h3>
<h3>ENTER YOUR REGISTER NO : 212224230248</h3>
<h3>EX. NO : 04</h3>
<h3>DATE : 30/09/2025</h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```c
%%cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H

// Initialize matrix with random float values
void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Matrix addition on CPU
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

// Result check between host and GPU results
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Mismatch at index %d: host %f vs gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
    else
    {
        printf("Arrays match.\n\n");
    }
}

// Matrix addition kernel on GPU
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    // Calculate thread's absolute index in 2D grid
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    // Check if we are within the matrix bounds
    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Set up matrix size
    int nx, ny;
    int ishift = 12;  // Default: 2^12 = 4096

    if (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;  // square matrix
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx = %d, ny = %d\n", nx, ny);

    // Allocate Unified Memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes));

    // Initialize data
    double iStart = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    double iElaps = seconds() - iStart;
    printf("Initialization:\t\t%f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Host computation
    iStart = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t%f sec\n", iElaps);

    // Kernel launch config
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Warm-up kernel
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
    CHECK(cudaDeviceSynchronize());

    // Actual kernel launch
    iStart = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on GPU:\t%f sec <<<(%d,%d), (%d,%d)>>>\n",
           iElaps, grid.x, grid.y, block.x, block.y);

    // Check for kernel errors
    CHECK(cudaGetLastError());

    // Compare results
    checkResult(hostRef, gpuRef, nxy);

    // Cleanup
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    CHECK(cudaDeviceReset());
    return 0;
}
```

## OUTPUT:
### with memSet

![alt text](<Screenshot 2025-09-30 093440.png>)

### without memSet

![alt text](<Screenshot 2025-09-30 093448.png>)
## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given less/more 0.000001 less time.
