#include "CCCuDMA.h"
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
#include "DMAUtils.h"
#include <Argus/Argus.h>
#include <deque>
#include "CCKernel.h"

#include "npp.h"

class CCCuDMA::Detail
{
public:
	Detail(CCCuDMAProps &_props) : props(_props)
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
		inputImageType = inputRawMetadata->getImageType();
		inputChannels = inputRawMetadata->getChannels();
		srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
		srcPitch[0] = static_cast<int>(inputRawMetadata->getStep()) >> 2;
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		outputImageType = outputRawMetadata->getImageType();
		outputChannels = outputRawMetadata->getChannels();
		dstPitch = static_cast<int>(outputRawMetadata->getStep());
        dstNextPtrOffset[0] = 0;

		return true;
	}

	bool rgbaSetMetadata(framemetadata_sp& input){
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		rgbaPitch = static_cast<int>(inputRawMetadata->getStep());
		rgbaRowSize = inputRawMetadata->getRowSize();
		rgbaHeight = inputRawMetadata->getHeight();
	}

	bool compute(void* buffer, DMAFDWrapper* rgbaDMAFdWrapper, DMAFDWrapper* outDMAFdWrapper)
	{
		auto rgbaBuffer = rgbaDMAFdWrapper->cudaPtr;
        auto dstPtr = outDMAFdWrapper->cudaPtr;
        for(auto i = 0; i < outputChannels; i++)
		{
			src[i] = static_cast<Npp32f*>(buffer) + dstNextPtrOffset[i];
		}

		cudaMemcpy2DAsync(dstPtr, dstPitch, rgbaBuffer, rgbaPitch, rgbaRowSize, rgbaHeight, cudaMemcpyDeviceToDevice, props.stream);

        lanuchColorMapMONO2RGBA(src[0], srcPitch[0], dstPtr + srcSize[0].width * 4, dstPitch, srcSize[0], props.stream);

		return true;
	}

    EGLDisplay eglDisplay;
private:
	int rgbaPitch, rgbaHeight;
	size_t rgbaRowSize;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;
	int inputChannels;
	int outputChannels;
    Npp32f* src[4];
	NppiSize srcSize[4];
	int srcPitch[4];
	NppiSize dstSize[4];
	int dstPitch;
    size_t dstNextPtrOffset[4];

	CCCuDMAProps props;
};

CCCuDMA::CCCuDMA(CCCuDMAProps _props) : Module(TRANSFORM, "CCCuDMA", _props), props(_props), mFrameChecker(0)
{
	mDetail.reset(new Detail(_props));	
}

CCCuDMA::~CCCuDMA() {}

bool CCCuDMA::validateInputPins()
{
	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE && memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE or DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool CCCuDMA::validateOutputPins()
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

void CCCuDMA::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	switch (inputRawMetadata->getImageType())
	{
		case ImageMetadata::RGBA:
			break;
		case ImageMetadata::MONO:
			if(!mOutputMetadata){
				mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
				mOutputMetadata->copyHint(*metadata.get());
				mOutputPinId = addOutputPin(mOutputMetadata);
			}
			break;
		default:
			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
	}	
}

bool CCCuDMA::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool CCCuDMA::term()
{
	return Module::term();
}

bool CCCuDMA::process(frame_container &frames)
{
	cudaFree(0);
	frame_sp rgba_frame, mono_frame;
	for (auto const& element : frames)
	{
		auto frame = element.second;
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frame->getMetadata());
		if(rawOutMetadata->getImageType() == ImageMetadata::ImageType::RGBA){
			rgba_frame = frame;
		}else{
			mono_frame = frame;
		}
	}
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);

    mDetail->compute(mono_frame->data(),(static_cast<DMAFDWrapper *>(rgba_frame->data())) ,(static_cast<DMAFDWrapper *>(outFrame->data())));

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCCuDMA::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	if(rawMetadata->getImageType() == ImageMetadata::MONO){
		setMetadata(metadata);
	}else{
		mDetail->rgbaSetMetadata(metadata);
	}
	mFrameChecker++;

	return true;
}

void CCCuDMA::setMetadata(framemetadata_sp& metadata)
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
	RawImageMetadata outputMetadata(2*width, height, props.imageType, CV_8UC4, 512, CV_8U, FrameMetadata::DMABUF, true);		
	rawOutMetadata->setData(outputMetadata);
	mDetail->setMetadata(metadata, mOutputMetadata);
}

bool CCCuDMA::shouldTriggerSOS()
{
	return (mFrameChecker == 0 || mFrameChecker == 1);
}

bool CCCuDMA::processEOS(string& pinId)
{
	mFrameChecker = 0;
	return true;
}