#include "launchDivision.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"

__global__ void applaunchDivision(const Npp8u* src, int nSrcStep, Npp32f* dst_rgb, int rDstStep, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y_ = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y_ >= height)
	{
		return;
	}

	int dst_offset = (y_ * rDstStep +  ((3 * x) << 2) );
	int offset = y_* nSrcStep + ((3*x) << 2);

	#pragma unroll
	for(auto i = 0; i < 4; i++)
	{
			dst_rgb[dst_offset] = src[offset] / 255.0;
			dst_rgb[dst_offset+1] = src[offset+1] / 255.0;
			dst_rgb[dst_offset+2] = src[offset+2] / 255.0;

		offset += 3;
		dst_offset += 3;
	}
}

void launchDivision(const Npp8u* src,int nSrcStep, Npp32f* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream){
	dim3 block(32, 32); 
	dim3 grid(((oSizeROI.width >> 2) + block.x - 1) / (1 * block.x), (oSizeROI.height + block.y - 1) / block.y);
	applaunchDivision <<<grid, block, 0, stream>>> (src, nSrcStep, dst, rDstStep, oSizeROI.width, oSizeROI.height);
}

__global__ void applaunchMul(const Npp32f* src, int nSrcStep, Npp8u* dst_rgb, int rDstStep, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y_ = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y_ >= height)
	{
		return;
	}

	int dst_offset = (y_ * rDstStep +  ((3 * x) << 2) );
	int offset = y_* nSrcStep + ((3*x) << 2);

	#pragma unroll
	for(auto i = 0; i < 4; i++)
	{
			dst_rgb[dst_offset] = src[offset] * 255.0;
			dst_rgb[dst_offset+1] = src[offset+1] * 255.0;
			dst_rgb[dst_offset+2] = src[offset+2] * 255.0;

		offset += 3;
		dst_offset += 3;
	}
}

void launchMul(const Npp32f* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream){
	dim3 block(32, 32); 
	dim3 grid(((oSizeROI.width >> 2) + block.x - 1) / (1 * block.x), (oSizeROI.height + block.y - 1) / block.y);
	applaunchMul <<<grid, block, 0, stream>>> (src, nSrcStep, dst, rDstStep, oSizeROI.width, oSizeROI.height);
}