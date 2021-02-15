#include "DMAUtilWrapper.h"
#include "Logger.h"

DMAUtilWrapper *DMAUtilWrapper::create(uint32_t height,uint32_t width,
                                    NvBufferColorFormat colorFormat,
                                    NvBufferLayout layout, EGLDisplay eglDisplay)
{
    DMAUtilWrapper *buffer = new DMAUtilWrapper(height,width);
    if (!buffer)
    {
        return nullptr;
    }

    NvBufferCreateParams inputParams = {0};

    inputParams.width = width;
    inputParams.height = height;
    inputParams.layout = layout;
    inputParams.colorFormat = colorFormat;
    inputParams.payloadType = NvBufferPayload_SurfArray;
    inputParams.nvbuf_tag = NvBufferTag_CAMERA;

    if (NvBufferCreateEx(&buffer->m_fd, &inputParams))
    {
        LOG_ERROR << "Failed NvBufferCreateEx";
        delete buffer;
        return nullptr;
    }

    buffer->eglImage = NvEGLImageFromFd(eglDisplay, buffer->m_fd);
    if (buffer->eglImage == EGL_NO_IMAGE_KHR)
    {
        LOG_ERROR << "Failed to create EGLImage";
        delete buffer;
        return nullptr;
    }

    return buffer;
}

DMAUtilWrapper::DMAUtilWrapper(uint32_t height,uint32_t width) : m_buffer(nullptr), eglImage(EGL_NO_IMAGE_KHR), m_fd(-1)
{
}

DMAUtilWrapper::~DMAUtilWrapper()
{
    if (eglImage != EGL_NO_IMAGE_KHR)
    {
        auto res = NvDestroyEGLImage(NULL, eglImage);
        if(res)
        {
            LOG_ERROR << "NvDestroyEGLImage Error<>" << res;
        }
    }

    if (m_fd >= 0)
    {
        NvBufferDestroy(m_fd);
        m_fd = -1;
    }
}