#include "CMHost.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"


class CMHost::Detail
{
public:
	Detail(CMHostProps &_props) : props(_props)
	{
	}

	~Detail()
	{

	}

	bool compute(void* buffer, void* outFrame)
	{
        cv::Mat temp(1024, 1024, CV_32FC1, buffer);
		cv::Mat dest8u(1024, 1024, CV_8U);
        cv::Mat dest(1024, 1024, CV_8UC3, outFrame);
		temp = 255*temp;
        temp.convertTo(dest8u, CV_8U);
        cv::applyColorMap(dest8u,dest, cv::COLORMAP_JET);

		return true;
	}
private:

	CMHostProps props;
};

CMHost::CMHost(CMHostProps _props) : Module(TRANSFORM, "CMHost", _props), props(_props), mFrameLength(0)
{
	mDetail.reset(new Detail(_props));	
}

CMHost::~CMHost() {}

bool CMHost::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool CMHost::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	auto mOutputFrameType = metadata->getFrameType();
	if (mOutputFrameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << mOutputFrameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void CMHost::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::HOST));
	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool CMHost::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CMHost::term()
{
	return Module::term();
}

bool CMHost::process(frame_container &frames)
{
	cudaFree(0);
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame();

    mDetail->compute(frame->data(),outFrame->data());

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CMHost::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void CMHost::setMetadata(framemetadata_sp& metadata)
{
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	RawImageMetadata outputMetadata(width, height, ImageMetadata::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);		
	rawOutMetadata->setData(outputMetadata);
	

	mFrameLength = mOutputMetadata->getDataSize();
}

bool CMHost::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CMHost::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}