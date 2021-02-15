#pragma once
#include "Allocators.h"
#include "nvbuf_utils.h"
#include "DMAUtilWrapper.h"
#include <deque>

class DMAAllocator : public HostAllocator
{
private:
    deque<DMAFDWrapper *> dmaFD;
    int freeDmaCount;
    // deque of DMAFD
    // freebuffer count

public:
    DMAAllocator(framemetadata_sp &framemetadata) : buff_allocator(APRA_CHUNK_SZ), freeDmaCount(10)
    {
        for (auto i = 0; i < freeDmaCount; i++)
        {
            dmaFDWrapper = DMABuffer::create(streamSize, NvBufferColorFormat_UYVY, NvBufferLayout_BlockLinear, eglDisplay);
            if (!dmaFDWrapper)
            {
                LOG_ERROR << "Failed to allocate dmaFDWrapper";
                return false;
            }
            dmaFD.push_back(dmaFDWrapper);
        }
    };
    ~DMAAllocator()
    {
        while (dmaFD)
        {
            DMAFDWrapper *dmaFDWrapper = dmaFD.front();
            dmaFD.pop_front();
            delete dmaFDWrapper;
            dmaFDWrapper = nullptr;
        }
        // loop through your dq and destruction of DMAFD
    }
    void *allocateChunks(size_t n)
    {
        DMAFDWrapper *dmaFDWrapper;
        if (freeDmaCount == 0)
        {
            dmaFDWrapper = dmaFD.front();
            dmaFD.pop_front();
            freeDmaCount--;
        }
        return dmaFDWrapper;
        // if freecount = 0
        // allocate
        // and send 1 free buffer by pop
        // reduce freecount
    }

    void freeChunks(void *MemPtr, size_t n)
    {
        dmaFD.push_back(MemPtr);
        freeDmaCount++;
        // push
        // inc freecount
    }
};