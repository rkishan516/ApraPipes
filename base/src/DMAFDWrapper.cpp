#include "DMAFDWrapper.h"
#include "nvbuf_utils.h"
#include "Logger.h"

DMAFDWrapper *DMAFDWrapper::create(int index, int width, int height,
                                    NvBufferColorFormat colorFormat,
                                    NvBufferLayout layout, EGLDisplay eglDisplay)
{
    DMAFDWrapper *buffer = new DMAFDWrapper(index);
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

    return buffer;
}

DMAFDWrapper::DMAFDWrapper(int _index) : eglImage(EGL_NO_IMAGE_KHR), m_fd(-1), index(_index)
{
}

DMAFDWrapper::~DMAFDWrapper()
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