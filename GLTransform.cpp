#include "GLTransform.h"
#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "CCKernel.h"
#include "DMAFDWrapper.h"
#include "DMABuffer.h"
#include <Argus/Argus.h>
#include <deque>

#include "npp.h"

class GLTransform::Detail
{
public:
	Detail(GLTransformProps &_props) : props(_props)
	{
        dest_rect.top = _props.top;
        dest_rect.left = _props.left;
        dest_rect.width = _props.width;
        dest_rect.height = _props.height;   
		memset(&transParams, 0, sizeof(transParams));
		transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
		transParams.transform_filter = NvBufferTransform_Filter_Smart;
		transParams.dst_rect = dest_rect;
	}

	~Detail()
	{

	}

	bool compute(frame_sp& frame, int outFD)
	{
		auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());
		NvBufferTransform(dmaFDWrapper->getFd(), outFD, &transParams);

		return true;
	}

public:
	NvBufferRect dest_rect;
	framemetadata_sp outputMetadata;
	std::string outputPinId;
	GLTransformProps props;

private:	
	NvBufferTransformParams transParams;
};

GLTransform::GLTransform(GLTransformProps props) : Module(TRANSFORM, "GLTransform", props)
{
	mDetail.reset(new Detail(props));	
}

GLTransform::~GLTransform() {}

bool GLTransform::validateInputPins()
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

bool GLTransform::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();	
	if (metadata->getFrameType() != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << metadata->getFrameType() << ">";
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

void GLTransform::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	switch (mDetail->props.imageType)
	{
		case ImageMetadata::RGBA:
			mDetail->outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
			break;
		default:
			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
	}

	mDetail->outputMetadata->copyHint(*metadata.get());
	mDetail->outputPinId = addOutputPin(mDetail->outputMetadata);	
}

bool GLTransform::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool GLTransform::term()
{
	return Module::term();
}

bool GLTransform::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame(mDetail->outputMetadata->getDataSize(),mDetail->outputPinId);
	if(!outFrame.get())
	{
		LOG_ERROR << "FAILED TO GET BUFFER";
		return false;
	}

	auto dmaFdWrapper = static_cast<DMAFDWrapper *>(outFrame->data());
	dmaFdWrapper->tempFD = dmaFdWrapper->getFd();

    mDetail->compute(frame,dmaFdWrapper->tempFD);

	frames.insert(make_pair(mDetail->outputPinId, outFrame));
	send(frames);

	return true;
}

bool GLTransform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void GLTransform::setMetadata(framemetadata_sp& metadata)
{	
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	
	if (!(mDetail->props.imageType == ImageMetadata::RGBA && inputImageType == ImageMetadata::RGBA))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Color conversion not supported");
	}
	
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->outputMetadata);
	RawImageMetadata outputMetadata(mDetail->props.width, mDetail->props.height, mDetail->props.imageType, CV_8UC4, mDetail->props.width*4 , CV_8U, FrameMetadata::DMABUF, true);		
	rawOutMetadata->setData(outputMetadata);
}

bool GLTransform::processEOS(string& pinId)
{
	mDetail->outputMetadata.reset();
	return true;
}