#pragma once

#include "nppdefs.h"

void launchDivision(const Npp8u* src,int nSrcStep, Npp32f* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream);
void launchMul(const Npp32f* src,int nSrcStep, Npp8u* dst,int rDstStep, NppiSize oSizeROI, cudaStream_t stream);