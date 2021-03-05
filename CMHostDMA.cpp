#include "CMHostDMA.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"


class CMHostDMA::Detail
{
public:
	Detail(CMHostDMAProps &_props) : props(_props)
	{
	}

	~Detail()
	{

	}

	bool compute(void* buffer, void* outFrame)
	{
        cv::Mat temp(1024, 1024, CV_32FC1, buffer);
        cv::Mat dest(1024, 1024, CV_8UC3, outFrame);
		temp = 255*temp;
        temp.convertTo(dest, CV_8U);
        cv::applyColorMap(dest,dest, cv::COLORMAP_JET);

		return true;
	}
private:

	CMHostDMAProps props;
};

CMHostDMA::CMHostDMA(CMHostDMAProps _props) : Module(TRANSFORM, "CMHostDMA", _props), props(_props), mFrameLength(0)
{
	mDetail.reset(new Detail(_props));	
}

CMHostDMA::~CMHostDMA() {}

bool CMHostDMA::validateInputPins()
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

bool CMHostDMA::validateOutputPins()
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void CMHostDMA::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	switch (props.imageType)
	{
		case ImageMetadata::RGBA:
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
			break;
		default:
			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool CMHostDMA::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CMHostDMA::term()
{
	return Module::term();
}

bool CMHostDMA::process(frame_container &frames)
{
	cudaFree(0);
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);
    auto tempFrame = (static_cast<DMAFDWrapper *>(outFrame->data()))->hostPtr;

    mDetail->compute(frame->data(),tempFrame);

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CMHostDMA::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void CMHostDMA::setMetadata(framemetadata_sp& metadata)
{
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	if (!(props.imageType == ImageMetadata::RGBA && inputImageType == ImageMetadata::MONO))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Color conversion not supported");
	}
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	RawImageMetadata outputMetadata(width, height, props.imageType, CV_8UC4, 512, CV_8U, FrameMetadata::DMABUF, true);		
	rawOutMetadata->setData(outputMetadata);
	

	mFrameLength = mOutputMetadata->getDataSize();
}

bool CMHostDMA::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CMHostDMA::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}