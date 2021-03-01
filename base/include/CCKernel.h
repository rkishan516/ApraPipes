#pragma once

#include "nppdefs.h"

void lanuchAPPYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst[3], int rDstStep, NppiSize oSizeROI, cudaStream_t stream);

void lanuchAPPUYVYToBGRA(const Npp8u* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI);

void lanuchAPPRGBAToRGB(const Npp8u* src,int nSrcStep, Npp32f* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream);

void lanuchColorMapMONO2RGBA(const Npp32f* src, int nSrcStep, Npp8u* dst, int rDstStep, NppiSize oSizeROI, cudaStream_t stream);