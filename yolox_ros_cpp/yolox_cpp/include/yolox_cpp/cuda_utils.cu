
__global__ void blobFromImage(uchar3* image_data, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int uchar3_idx = y * width + x;

    int b_idx = 0 * height * width + y * width + x;
    int g_idx = 1 * height * width + y * width + x;
    int r_idx = 2 * height * width + y * width + x;

    output[b_idx] = image_data[uchar3_idx].x;
    output[g_idx] = image_data[uchar3_idx].y;
    output[r_idx] = image_data[uchar3_idx].z;

    return;
}

void launchBlobFromImage(uchar3* d_input, float* d_output, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    blobFromImage<<<grid, block, 0, stream>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(stream);
}

__global__ void generate_grids_and_strides() {
    
}