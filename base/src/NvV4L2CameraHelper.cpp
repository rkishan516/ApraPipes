#include "DMAFDWrapper.h"
#include "NvV4L2CameraHelper.h"
#include "NvEglRenderer.h"
#include "NvUtils.h"
#include "nvbuf_utils.h"
#include "Logger.h"

NvV4L2CameraHelper::NvV4L2CameraHelper()
{
    camDevname = "/dev/video0";
    camFD = -1;
    camPixFmt = V4L2_PIX_FMT_UYVY;

    mRunning = false;
}

NvV4L2CameraHelper::~NvV4L2CameraHelper()
{
    LOG_INFO << "in destructor ------------------";

    if (camFD > 0){
        close(camFD);
    }

    LOG_INFO << "out of destructor ------------------";
}

std::shared_ptr<NvV4L2CameraHelper> NvV4L2CameraHelper::create(SendFrame sendFrame,std::function<frame_sp()> _makeFrame)
{
    auto instance = std::make_shared<NvV4L2CameraHelper>();
    instance->mSendFrame = sendFrame;
    instance->makeFrame = _makeFrame;

    return instance;
}

bool NvV4L2CameraHelper::camera_initialize()
{
    struct v4l2_format fmt;

    /* Open camera device */
    camFD = open(camDevname, O_RDWR);
    if (camFD == -1)
    {
        LOG_ERROR << "Failed to open camera /dev/video0";
        return false;
    }

    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = camWidth;
    fmt.fmt.pix.height = camHeight;
    fmt.fmt.pix.pixelformat = camPixFmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(camFD, VIDIOC_S_FMT, &fmt) < 0)
    {
        LOG_ERROR << "Failed to set camera ouput format to UYVY";
        return false;
    }

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camFD, VIDIOC_G_FMT, &fmt) < 0)
    {
        LOG_ERROR << "Failed to get camera output format";
        return false;
    }

    if (fmt.fmt.pix.width != camWidth ||
        fmt.fmt.pix.height != camHeight ||
        fmt.fmt.pix.pixelformat != camPixFmt)
    {
        LOG_WARNING << "The desired format is not supported";
        LOG_ERROR << "Supported width is : " << fmt.fmt.pix.width;
        LOG_ERROR << "Supported height is : " << fmt.fmt.pix.height;
        LOG_ERROR << "Supported pixelformat is : " << fmt.fmt.pix.pixelformat;

        return false;
    }

    return true;
}

bool NvV4L2CameraHelper::start_stream()
{
    enum v4l2_buf_type type;

    /* Start v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camFD, VIDIOC_STREAMON, &type) < 0)
    {
        LOG_ERROR << "Failed to start streaming";
        return false;
    }

    LOG_INFO << "Camera video streaming on ...";
    return true;
}

bool NvV4L2CameraHelper::stop_stream()
{
    enum v4l2_buf_type type;

    /* Stop v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camFD, VIDIOC_STREAMOFF, &type))
    {
        LOG_ERROR << "Failed to stop streaming";
        return false;
    }

    LOG_INFO << "Camera video streaming off ...";
    return true;
}

void NvV4L2CameraHelper::operator()()
{
    int fds;
    fd_set rset;

    fds = camFD;
    FD_ZERO(&rset);
    FD_SET(fds, &rset);
    mRunning = true;

    /* Wait for camera event with timeout = 5000 ms */
    while (select(fds + 1, &rset, NULL, NULL, NULL) > 0 && mRunning)
    {
        if (FD_ISSET(fds, &rset))
        {
            struct v4l2_buffer v4l2_buf;

            /* Dequeue a camera buff */
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            if (ioctl(camFD, VIDIOC_DQBUF, &v4l2_buf) < 0)
            {
                LOG_ERROR << "Failed to dequeue camera buff";
                break;
            }
            auto frameItr = bufferFD.find(v4l2_buf.m.fd);            
            mSendFrame(frameItr->second);
            bufferFD.erase(frameItr);
        }
    }
}

bool NvV4L2CameraHelper::queueBufferToCamera()
{
    while(true)
    {
        auto frame = makeFrame();
        if(!frame.get()){
            break;
        }
        auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());

        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = dmaFDWrapper->index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(camFD, VIDIOC_QUERYBUF, &buf) < 0){
            LOG_ERROR << "Failed to query buff";
            return false;
        }
        buf.m.fd = (unsigned long)dmaFDWrapper->tempFD;

        bufferFD.insert(make_pair(buf.m.fd, frame));
        
        if (ioctl(camFD, VIDIOC_QBUF, &buf) < 0){
            LOG_ERROR << "Failed to enqueue buffers";
            return false;
        }
    }
    return true;
}

bool NvV4L2CameraHelper::prepare_buffers()
{
    if (!request_camera_buff())
    {
        LOG_ERROR << "Failed to set up camera buffer";
        return false;
    }   

    LOG_INFO << "Succeed in preparing stream buffers";
    return true;
}

bool NvV4L2CameraHelper::request_camera_buff()
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = maxConcurrentFrames;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(camFD, VIDIOC_REQBUFS, &rb) < 0)
    {
        LOG_ERROR << "Failed to request v4l2 buffers";
        return false;
    }

    if (rb.count != maxConcurrentFrames)
    {
        LOG_ERROR << "V4l2 buffer number is not as desired";
        return false;
    }


    if(!queueBufferToCamera()){
        return false; 
    }
    
    return true;
}

bool NvV4L2CameraHelper::start(uint32_t width, uint32_t height, uint32_t _maxConcurrentFrames)
{
    camHeight = height;
    camWidth = width;
    maxConcurrentFrames = _maxConcurrentFrames;
    bool status = false;
    status = camera_initialize();
    if (status == false)
    {
        LOG_ERROR << "Camera Initialization Failed";
        return false;
    }
    status = prepare_buffers();
    if (status == false)
    {
        LOG_ERROR << "Buffer Preparation Failed";
        return false;
    }
    status = start_stream();
    if (status == false)
    {
        LOG_ERROR << "Start Stream Failed";
        return false;
    }
    mThread = std::thread(std::ref(*this));

    return true;
}

bool NvV4L2CameraHelper::stop()
{
    LOG_INFO << "STOP SIGNAL STARTING";

    mRunning = false;

    if (!stop_stream())
    {
        LOG_ERROR << "Stop Stream Failed";
        return false;
    }

    mThread.join();

    LOG_INFO << "Coming out of stop helper";

    return true;
}