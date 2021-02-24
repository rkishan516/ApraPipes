#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CCKernel.h"

__global__ void appYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst_y, Npp8u* dst_u, Npp8u* dst_v, int rDstStep, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y >= height)
	{
		return;
	}	

	int x_ = 4 * x;
	int dst_offset = y * rDstStep + x_;
	int offset = y* nSrcStep + 6 * x;

	dst_y[dst_offset + 0] = src[offset + 1];
	dst_y[dst_offset + 1] = src[offset + 2];
	dst_y[dst_offset + 2] = src[offset + 4];
	dst_y[dst_offset + 3] = src[offset + 5];
	
	
	auto u_value = src[offset];
	dst_u[dst_offset + 0] = u_value;
	dst_u[dst_offset + 1] = u_value;
	dst_u[dst_offset + 2] = u_value;
	dst_u[dst_offset + 3] = u_value;

	auto v_value = src[offset + 3];
	dst_v[dst_offset + 0] = v_value;
	dst_v[dst_offset + 1] = v_value;
	dst_v[dst_offset + 2] = v_value;
	dst_v[dst_offset + 3] = v_value;
}

__global__ void appUYVYToBGR(const Npp8u* src, int nSrcStep, Npp8u* dst_bgr, int rDstStep, int width, int height, bool is_alpha)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y_ = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y_ >= height)
	{
		return;
	}

	int dst_offset = y_ * rDstStep + 6 * x;
	if(is_alpha){
		// for bgra we have 8 steps for one uyvy
		dst_offset += 2 * x;
	}
	int offset = y_* nSrcStep + 4 * x;

	Npp8u y = *(src+offset);src++;
	Npp8u u = *(src+offset);src++;
	Npp8u v = *(src+offset);src++;

	// https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/3fc53111-3152-4c12-9fa5-e4bbb666ae9a/how-to-change-uyvy-to-rgb?forum=windowsdirectshowdevelopment
	*(dst_bgr+dst_offset) = static_cast<Npp8u>(1.0*y + 1.772*(u-128));dst_bgr++;
	*(dst_bgr+dst_offset) = static_cast<Npp8u>(y+0.34413*(u-128)-0.71414*(v-128));dst_bgr++;
	*(dst_bgr+dst_offset) = static_cast<Npp8u>(y+8+1.402*(v-128));dst_bgr++;
	if(is_alpha){
		*(dst_bgr+dst_offset) = static_cast<Npp8u>(255);dst_bgr++;
	}

	y = *(src+offset);

	*(dst_bgr+dst_offset) = static_cast<Npp8u>(1.0*y + 1.772*(u-128));dst_bgr++;
	*(dst_bgr+dst_offset) = static_cast<Npp8u>(y+0.34413*(u-128)-0.71414*(v-128));dst_bgr++;
	*(dst_bgr+dst_offset) = static_cast<Npp8u>(y+8+1.402*(v-128));
	if(is_alpha){
		dst_bgr++;
		*(dst_bgr+dst_offset) = static_cast<Npp8u>(255);
	}
}

__global__ void appRGBAToRGB(const Npp8u* src, int nSrcStep, Npp8u* dst_rgb, int rDstStep, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y_ = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y_ >= height)
	{
		return;
	}

	int dst_offset = y_ * rDstStep + 3 * x;
	int offset = y_* nSrcStep + 4 * x;

	*(dst_rgb+dst_offset) = *(src+offset);
	*(dst_rgb+dst_offset+1) = *(src+offset+1);
	*(dst_rgb+dst_offset+2) = *(src+offset+2);
}

void lanuchAPPYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst[3], int rDstStep, NppiSize oSizeROI, cudaStream_t stream)
{
	dim3 block(32, 32); 
	dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
	appYUV411ToYUV444 <<<grid, block, 0, stream>>> (src, nSrcStep, dst[0], dst[1], dst[2], rDstStep, oSizeROI.width >> 2, oSizeROI.height);
}

void lanuchAPPUYVYToBGR(const Npp8u* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream){
	dim3 block(32, 32); 
	dim3 grid((oSizeROI.width + block.x - 1) / (2 * block.x), (oSizeROI.height + block.y - 1) / block.y);
	appUYVYToBGR <<<grid, block, 0, stream>>> (src, nSrcStep, dst, rDstStep, oSizeROI.width >> 2, oSizeROI.height,false);
}

void lanuchAPPUYVYToBGRA(const Npp8u* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI){
	dim3 block(32, 32); 
	dim3 grid((oSizeROI.width + block.x - 1) / (2 * block.x), (oSizeROI.height + block.y - 1) / block.y);
	appUYVYToBGR <<<grid, block, 0>>> (src, nSrcStep, dst, rDstStep, oSizeROI.width >> 2, oSizeROI.height,true);
}

void lanuchAPPRGBAToRGB(const Npp8u* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream){
	dim3 block(32, 32); 
	dim3 grid((oSizeROI.width + block.x - 1) / (1 * block.x), (oSizeROI.height + block.y - 1) / block.y);
	appRGBAToRGB <<<grid, block, 0, stream>>> (src, nSrcStep, dst, rDstStep, oSizeROI.width >> 2, oSizeROI.height);
}