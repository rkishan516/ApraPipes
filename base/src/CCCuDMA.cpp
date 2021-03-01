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
		srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		outputImageType = outputRawMetadata->getImageType();
		outputChannels = outputRawMetadata->getChannels();
		dstPitch = static_cast<int>(outputRawMetadata->getStep());
        dstNextPtrOffset[0] = 0;

		return true;
	}

	bool compute(void* buffer, int outFD)
	{
        auto dstPtr = DMAUtils::getCudaPtrForFD(outFD, eglOutImage,&pOutResource,eglOutFrame, eglDisplay);
        for(auto i = 0; i < outputChannels; i++)
		{
			src[i] = static_cast<Npp32f*>(buffer) + dstNextPtrOffset[i];
		}

        lanuchColorMapMONO2RGBA(src[0], srcPitch[0], dstPtr, dstPitch, srcSize[0], props.stream);

        DMAUtils::freeCudaPtr(eglOutImage,&pOutResource, eglDisplay);

		return true;
	}

    EGLDisplay eglDisplay;
private:
	EGLImageKHR eglOutImage;
    CUgraphicsResource pOutResource;
    CUeglFrame eglOutFrame;
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

CCCuDMA::CCCuDMA(CCCuDMAProps _props) : Module(TRANSFORM, "CCCuDMA", _props), props(_props), mFrameLength(0), mNoChange(false)
{
	mDetail.reset(new Detail(_props));	
}

CCCuDMA::~CCCuDMA() {}

bool CCCuDMA::validateInputPins()
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
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
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
	mOutputFrameType = metadata->getFrameType();
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
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);
    auto outFd = (static_cast<DMAFDWrapper *>(outFrame->data()))->tempFD;

    mDetail->compute(frame->data(),outFd);

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCCuDMA::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void CCCuDMA::setMetadata(framemetadata_sp& metadata)
{
	mInputFrameType = metadata->getFrameType();

	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
	int type = NOT_SET_NUM;
	int depth = NOT_SET_NUM;	
	ImageMetadata::ImageType inputImageType = ImageMetadata::MONO;

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	width = rawMetadata->getWidth();
	height = rawMetadata->getHeight();
	type = rawMetadata->getType();
	depth = rawMetadata->getDepth();
	inputImageType = rawMetadata->getImageType();

	mNoChange = false;
	if (inputImageType == props.imageType)
	{
		mNoChange = true;
		return;
	}

	if (!(props.imageType == ImageMetadata::RGBA && inputImageType == ImageMetadata::MONO))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Color conversion not supported");
	}
	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	RawImageMetadata outputMetadata(width, height, props.imageType, CV_8UC4, 512, CV_8U, FrameMetadata::DMABUF, true);		
	rawOutMetadata->setData(outputMetadata);
	

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata);	
}

bool CCCuDMA::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CCCuDMA::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}