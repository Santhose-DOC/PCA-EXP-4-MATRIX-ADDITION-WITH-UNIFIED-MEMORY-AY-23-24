%%cu
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