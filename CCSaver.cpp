#include "CCSaver.h"
#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "launchDivision.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include <Argus/Argus.h>
#include <deque>

#include "npp.h"

class CCSaver::Detail
{
public:
	Detail(CCSaverProps &_props) : props(_props)
	{
        eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if(eglDisplay == EGL_NO_DISPLAY)
        {
            throw AIPException(AIP_FATAL, "eglGetDisplay failed");
        } 

        if (!eglInitialize(eglDisplay, NULL, NULL))
        {
        throw AIPException(AIP_FATAL, "eglInitialize failed");
        }
	}

	~Detail()
	{

	}

	bool setMetadata(framemetadata_sp& input, framemetadata_sp& output)
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		resolution = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
		srcStep = inputRawMetadata->getStep();
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		dstStep = outputRawMetadata->getStep();

		return true;
	}

	bool compute(frame_sp& frame, void* buffer)
	{
        auto srcPtr = (static_cast<DMAFDWrapper *>(frame->data())->cudaPtr);
        
		auto dst = static_cast<Npp8u*>(buffer);

        lanuchAPPRGBAToRGBSaver(srcPtr, srcStep, dst, dstStep, resolution, props.stream);

		return true;
	}

    EGLDisplay eglDisplay;
private:
	size_t srcStep;
	size_t dstStep;
	NppiSize resolution;
	
	CCSaverProps props;
};

CCSaver::CCSaver(CCSaverProps _props) : Module(TRANSFORM, "CCSaver", _props), props(_props), mFrameLength(0)
{
	mDetail.reset(new Detail(_props));	
}

CCSaver::~CCSaver() {}

bool CCSaver::validateInputPins()
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

bool CCSaver::validateOutputPins()
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
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void CCSaver::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	switch (props.imageType)
	{
		case ImageMetadata::RGB:
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
			break;
		default:
			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool CCSaver::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CCSaver::term()
{
	return Module::term();
}

bool CCSaver::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);

    mDetail->compute(frame,outFrame->data());

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCSaver::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void CCSaver::setMetadata(framemetadata_sp& metadata)
{
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
	RawImageMetadata outputMetadata(width, height, props.imageType, CV_8UC3, 512, CV_8U, FrameMetadata::CUDA_DEVICE, true);		
	rawOutMetadata->setData(outputMetadata);

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata);
}

bool CCSaver::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CCSaver::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}