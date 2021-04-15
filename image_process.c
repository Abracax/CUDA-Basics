// Histogram Equalization
// Cast the image to unsigned char

// Convert the image from RGB to Gray Scale. You will find one of the lectures and textbook chapters helpful.

// Compute the histogram of the image

// Compute the scan (prefix sum) of the histogram to arrive at the histogram equalization function

// Apply the equalization function to the input image to get the color corrected image

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 32
#define BLOCK_SIZE 128

//@@ insert code here

__global__ void float_to_uchar(float *input, uint8_t *output, int width, int height)
{
    int w = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int h = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    int idx = c * (width * height) + h * (width) + w;

    if (w < width && h < height)
        output[idx] = (uint8_t)(255 * input[idx]);
}

__global__ void rgb_to_greyScale(uint8_t *input, uint8_t *output, int width, int height)
{
    int w = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int h = TILE_WIDTH * blockIdx.y + threadIdx.y;

    int idx = h * width + w;

    if (w < width && h < height)
    {
        uint8_t r = input[3 * idx];
        uint8_t g = input[3 * idx + 1];
        uint8_t b = input[3 * idx + 2];
        output[idx] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
    }
}

__global__ void grey_to_hist(uint8_t *input, uint32_t *output, int width, int height)
{
    __shared__ uint32_t T[HISTOGRAM_LENGTH];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (idx < HISTOGRAM_LENGTH)
    {
        T[idx] = 0;
    }

    __syncthreads();
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w < width && h < height)
    {
        int i = h * (width) + w;
        uint8_t val = input[i];
        atomicAdd(&(output[val]), 1);
    }

    __syncthreads();
    if (idx < HISTOGRAM_LENGTH)
    {
        atomicAdd(&(output[idx]), T[idx]);
    }
    
}

__global__ void hist_to_cdf(uint32_t *input, float *output, int width, int height)
{

   __shared__ unsigned int T[HISTOGRAM_LENGTH];
    int i = threadIdx.x;
    T[i] = input[i];

    int stride = 1;
    while (stride < 2 * BLOCK_SIZE)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride = stride * 2;
    }

    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if ((index + stride) < 2 * BLOCK_SIZE)
            T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();
    output[i] = T[i] / ((float)(width * height));
}

__global__ void equalize(uint8_t *ucharImage, float *cdf, int width, int height, int chan)
{
    int w = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int h = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    if (w < width && h < height)
    {
        int idx = c * height * width + h * width + w;
        float equalized = 255 * (cdf[ucharImage[idx]] - cdf[0]) / (1.0 - cdf[0]);
        ucharImage[idx] = (uint8_t)min(max(equalized, 0.0), 255.0);
    }
}

__global__ void uchar_to_float(uint8_t *input, float *output, int width, int height, int chan)
{
    int w = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int h = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    int idx = c * height * width + h * width + w;

    if (w < width && h < height && c < chan)
        output[idx] = (float)(input[idx] / 255.0);
}

int main(int argc, char **argv)
{
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    //@@ Insert more code here
    uint8_t *deviceImageUChar;
    uint8_t *deviceImageGrayScale;
    uint32_t *deviceImageHistogram;
    float *deviceImageCDF;
    float *deviceInputImageData;
    float *deviceOutputImageData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceImageUChar, imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
    cudaMalloc((void **)&deviceImageGrayScale, imageWidth * imageHeight * sizeof(uint8_t));
    cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(uint32_t));
    cudaMemset((void *)deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(uint32_t));
    cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid = dim3(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), imageChannels);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    float_to_uchar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceImageUChar, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    dimGrid = dim3(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), 1);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    rgb_to_greyScale<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageGrayScale, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    dimGrid = dim3(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), 1);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    grey_to_hist<<<dimGrid, dimBlock>>>(deviceImageGrayScale, deviceImageHistogram, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    dimGrid = dim3(1, 1, 1);
    dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
    hist_to_cdf<<<dimGrid, dimBlock>>>(deviceImageHistogram, deviceImageCDF, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    dimGrid = dim3(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), imageChannels);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    equalize<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageCDF, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    dimGrid = dim3(ceil((1.0 * imageWidth) / TILE_WIDTH), ceil((1.0 * imageHeight) / TILE_WIDTH), imageChannels);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    uchar_to_float<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

    wbSolution(args, outputImage);

    cudaFree(deviceImageUChar);
    cudaFree(deviceImageGrayScale);
    cudaFree(deviceImageHistogram);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    //@@ insert code here
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
