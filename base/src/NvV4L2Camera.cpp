#include "NvV4L2Camera.h"
#include "FrameMetadata.h"

NvV4L2Camera::NvV4L2Camera(NvV4L2CameraProps props)
	: Module(SOURCE, "NvV4L2Camera", props)
{
	mOutputMetadata = framemetadata_sp(new RawImageMetadata(props.width, props.height, ImageMetadata::ImageType::UYVY, CV_8UC3, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
	mOutputPinId = addOutputPin(mOutputMetadata);

	mHelper = NvV4L2CameraHelper::create([&](frame_sp &frame) -> void {
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, frame));
		send(frames);
	},[&]() -> frame_sp {return makeFrame(mOutputMetadata->getDataSize(), mOutputPinId);});	
	height = props.height;
	width = props.width;
	maxConcurrentFrames = props.maxConcurrentFrames;
}

NvV4L2Camera::~NvV4L2Camera() {}

bool NvV4L2Camera::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		return false;
	}

	return true;
}

bool NvV4L2Camera::init()
{
	if (!Module::init())
	{
		return false;
	}
	mHelper->start(width, height,maxConcurrentFrames);

	return true;
}

bool NvV4L2Camera::term()
{
	auto ret = mHelper->stop();
	mHelper.reset();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}

bool NvV4L2Camera::produce()
{
	return mHelper->queueBufferToCamera();
}