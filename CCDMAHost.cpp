#include "CCDMAHost.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"

class CCDMAHost::Detail
{
public:
	Detail(CCDMAHostProps &_props) : props(_props)
	{
	}

	~Detail()
	{

	}

	bool compute(frame_sp& frame, void* buffer)
	{
        auto tempFrame = static_cast<DMAFDWrapper *>(frame->data())->hostPtr;

        cv::Mat temp(1024, 1024, CV_8UC4, tempFrame);
        cv::Mat dest(1024, 1024, CV_32FC3, buffer);
        cv::cvtColor(temp, temp ,cv::COLOR_RGBA2RGB);
        temp.convertTo(dest, CV_32FC3);
		dest = dest/255;
		return true;
	}

private:
	
	CCDMAHostProps props;
};

CCDMAHost::CCDMAHost(CCDMAHostProps _props) : Module(TRANSFORM, "CCDMAHost", _props), props(_props), mFrameLength(0), mNoChange(false)
{
	mDetail.reset(new Detail(_props));	
}

CCDMAHost::~CCDMAHost() {}

bool CCDMAHost::validateInputPins()
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool CCDMAHost::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	mOutputFrameType = metadata->getFrameType();
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

void CCDMAHost::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	switch (props.imageType)
	{
		case ImageMetadata::RGB:
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::HOST));
			break;
		default:
			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool CCDMAHost::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CCDMAHost::term()
{
	return Module::term();
}

bool CCDMAHost::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);

    mDetail->compute(frame,outFrame->data());

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCDMAHost::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void CCDMAHost::setMetadata(framemetadata_sp& metadata)
{
	mInputFrameType = metadata->getFrameType();

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	if (!(props.imageType == ImageMetadata::RGB && inputImageType == ImageMetadata::RGBA))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Color conversion not supported");
	}
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	RawImageMetadata outputMetadata(width, height, props.imageType, CV_32FC3, 512, CV_32F, FrameMetadata::HOST, true);		
	rawOutMetadata->setData(outputMetadata);

	mFrameLength = mOutputMetadata->getDataSize();
}

bool CCDMAHost::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CCDMAHost::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}