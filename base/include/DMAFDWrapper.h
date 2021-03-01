#pragma once
#include "nvbuf_utils.h"

class DMAFDWrapper
{
public:
    /* Always use this static method to create DMAFDWrapper */
    static DMAFDWrapper *create(int index, int width, int height,
                             NvBufferColorFormat colorFormat,
                             NvBufferLayout layout, EGLDisplay eglDisplay);

    virtual ~DMAFDWrapper();

    /* Return DMA buffer handle */
    int getFd() const { return m_fd; }
    EGLImageKHR getEGLImage() const { return eglImage; }

    int tempFD;
    int index;

private:
    DMAFDWrapper(int index);

    int m_fd;
    EGLImageKHR eglImage;
};