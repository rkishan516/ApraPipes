#pragma once

// #Mar10_Feedback - cleanup headers - keep only what is required - pool is not required - extframe ? - queue ?
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

public:
    NvV4L2CameraHelper();
    ~NvV4L2CameraHelper();
    static std::shared_ptr<NvV4L2CameraHelper> create(SendFrame sendFrame, std::function<frame_sp()> _makeFrame);

    bool start(uint32_t width, uint32_t height, uint32_t maxConcurrentFrames);
    bool stop();
    void operator()();
    bool queueBufferToCamera();

private:
    // #Mar10_Feedback - we are not using this - so you can remove this 
    void setSelf(std::shared_ptr<NvV4L2CameraHelper> &mother);
    std::thread mThread;
    // #Mar10_Feedback - consistency in variable names - use m before every variable - m means member - mBufferFDMutex
    std::mutex bufferFDMutex;

    // #Mar10_Feedback - consistency in variable names - use m before every variable 
    std::function<frame_sp()> makeFrame; 

    // #Mar10_Feedback - consistency in variable names - use m before every variable 
    /* Camera v4l2 context */
    const char * camDevname;
    int camFD;
    unsigned int camPixFmt;
    unsigned int camWidth;
    unsigned int camHeight;
    uint32_t maxConcurrentFrames;

    bool mRunning;
    SendFrame mSendFrame;
    // #Mar10_Feedback - consistency in variable names - use m before every variable 
    std::map<int, frame_sp> bufferFD;

    // #Mar10_Feedback - consistency in function names - camelcase - other functions are camelcase 
    bool camera_initialize();
    bool prepare_buffers();    
    bool start_stream();
    bool request_camera_buff();
    bool stop_stream();             
};