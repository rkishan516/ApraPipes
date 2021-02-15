#pragma once

#include "nvbuf_utils.h"

class DMAUtilWrapper
{
public:
    /* Always use this static method to create DMAUtilWrapper */
    static DMAUtilWrapper *create(const Argus::Size2D<uint32_t> &size,
                             NvBufferColorFormat colorFormat,
                             NvBufferLayout layout, EGLDisplay eglDisplay);

    virtual ~DMAUtilWrapper();

    /* Return DMA buffer handle */
    int getFd() const { return m_fd; }
    EGLImageKHR getEGLImage() const { return eglImage; }

    int tempFD;

private:
    DMAUtilWrapper(const Argus::Size2D<uint32_t> &size);

    int m_fd;
    EGLImageKHR eglImage;
};