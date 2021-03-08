#pragma once

#include <memory>
#include <thread>
#include <queue>
#include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>
#include "NvUtils.h"
#include "nvbuf_utils.h"
#include <map>
#include <mutex>

class NvV4L2CameraHelper
{
public:
    typedef std::function<void (frame_sp&)> SendFrame;
    NvV4L2CameraHelper();
    ~NvV4L2CameraHelper();
    static std::shared_ptr<NvV4L2CameraHelper> create(SendFrame sendFrame,std::function<frame_sp()> _makeFrame);

    bool start(uint32_t width, uint32_t height, uint32_t maxConcurrentFrames);
    bool stop();
    void operator()();
    bool queueBufferToCamera();
private:
    void setSelf(std::shared_ptr<NvV4L2CameraHelper> &mother);
    std::thread mThread;
    std::mutex bufferFDMutex;

    std::function<frame_sp()> makeFrame; 

    /* Camera v4l2 context */
    const char * camDevname;
    int camFD;
    unsigned int camPixFmt;
    unsigned int camWidth;
    unsigned int camHeight;
    uint32_t maxConcurrentFrames;


    bool mRunning;
    SendFrame mSendFrame;
    std::map<int, frame_sp> bufferFD;

    bool camera_initialize();
    bool prepare_buffers();    
    bool start_stream();
    bool request_camera_buff();
    bool stop_stream();             
};