#pragma once
#include "Allocators.h"
#include "DMAFDWrapper.h"
#include "FrameMetadataFactory.h"
#include <deque>

class DMAAllocator : public HostAllocator
{
private:
    std::deque<void *> dmaFD;
    std::vector<DMAFDWrapper*>  dmaFDWrapperArr;
    int freeDMACount;
    EGLDisplay eglDisplay;
    int height;
    int width;

public:
    DMAAllocator(framemetadata_sp framemetadata) : freeDMACount(0)
    {
        if(framemetadata->getFrameType() != FrameMetadata::FrameType::RAW_IMAGE){
            throw AIPException(AIP_FATAL, "Only Frame Type accepted are Raw Image");
        }

        eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if(eglDisplay == EGL_NO_DISPLAY)
        {
            throw AIPException(AIP_FATAL, "eglGetDisplay failed");
        } 
        
        if (!eglInitialize(eglDisplay, NULL, NULL))
        {
            throw AIPException(AIP_FATAL, "eglInitialize failed");
        } 
        
        if (framemetadata->getFrameType() == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(framemetadata);
			width = inputRawMetadata->getWidth();
            height = inputRawMetadata->getHeight();
		}

        // CHECK FOR IMAGE TYPE - SUPPORT ONLY UYVY
    };

    ~DMAAllocator()
    {
        for(auto wrapper : dmaFDWrapperArr)
        {
            delete wrapper;
        }

        eglTerminate(eglDisplay);
        if (!eglReleaseThread())
        {
            LOG_ERROR << "ERROR eglReleaseThread failed";
        }        
    }

    void *allocateChunks(size_t n)
    {
        if (freeDMACount == 0)
        {
            // remove hardcoding of UYVY
            auto dmaFDWrapper = DMAFDWrapper::create(width, height, NvBufferColorFormat_UYVY, NvBufferLayout_BlockLinear, eglDisplay);
            if (!dmaFDWrapper)
            {
                LOG_ERROR << "Failed to allocate dmaFDWrapper";
                throw AIPException(AIP_FATAL, "Memory Allocation Failed.");
            }
            dmaFDWrapperArr.push_back(dmaFDWrapper);
            freeDMACount++;
            dmaFD.push_back(static_cast<void*>(&dmaFDWrapper->tempFD));
        }
        
            auto fd = dmaFD.front();
            dmaFD.pop_front();
            freeDMACount--;
        

        return fd;
    }

    void freeChunks(void *MemPtr, size_t n)
    {
        dmaFD.push_back(MemPtr);
        freeDMACount++;
    }

    size_t getChunkSize()
    {
        return 1;
    }
};