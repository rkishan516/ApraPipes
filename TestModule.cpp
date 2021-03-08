#include "TestModule.h"
#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include <Argus/Argus.h>
#include <deque>
#include "launchDivision.h"

#include "npp.h"

class Test::Detail
{
public:
	Detail(TestProps &_props) : props(_props)
	{
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

	bool compute(void* buffer, void* dstPtr)
	{
		if(props.isDivision){
			src1 = static_cast<const Npp8u*>(buffer);
        	launchDivision(src1, srcPitch[0], static_cast<Npp32f*>(dstPtr), dstPitch >> 2, srcSize[0], props.stream);
		}else{
			for(auto i = 0; i < outputChannels; i++)
			{
				src[i] = static_cast<const Npp32f*>(buffer);
			}
			launchMul(src[0], srcPitch[0] >> 2, static_cast<Npp8u*>(dstPtr), dstPitch, srcSize[0], props.stream);
		}
		return true;
	}

private:
	int rgbaPitch, rgbaHeight;
	size_t rgbaRowSize;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;
	int inputChannels;
	int outputChannels;
    const Npp32f* src[4];
	const Npp8u* src1;
	NppiSize srcSize[4];
	int srcPitch[4];
	NppiSize dstSize[4];
	int dstPitch;
    size_t dstNextPtrOffset[4];

	TestProps props;
};

Test::Test(TestProps _props) : Module(TRANSFORM, "Test", _props), props(_props), mFrameChecker(0), mNoChange(false)
{
	mDetail.reset(new Detail(_props));	
}

Test::~Test() {}

bool Test::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE or CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool Test::validateOutputPins()
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
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void Test::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
    mOutputMetadata->copyHint(*metadata.get());
    mOutputPinId = addOutputPin(mOutputMetadata);
}

bool Test::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool Test::term()
{
	return Module::term();
}

bool Test::process(frame_container &frames)
{
	cudaFree(0);
	auto frame = frames.cbegin()->second;
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(),mOutputPinId);

    mDetail->compute(frame->data(),outFrame->data());

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool Test::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	setMetadata(metadata);
	mFrameChecker++;

	return true;
}

void Test::setMetadata(framemetadata_sp& metadata)
{
	mInputFrameType = metadata->getFrameType();

	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	if(props.isDivision){
		RawImageMetadata outputMetadata(width, height, ImageMetadata::RGB, CV_32FC3, width*3, CV_32F, FrameMetadata::CUDA_DEVICE, true);	
		rawOutMetadata->setData(outputMetadata);
		mDetail->setMetadata(metadata, mOutputMetadata);
	}
	else{
		RawImageMetadata outputMetadata(width, height, ImageMetadata::RGB, CV_8UC3, 512, CV_8U, FrameMetadata::CUDA_DEVICE, true);	
		rawOutMetadata->setData(outputMetadata);
		mDetail->setMetadata(metadata, mOutputMetadata);
	}
}

bool Test::shouldTriggerSOS()
{
	return (mFrameChecker == 0);
}

bool Test::processEOS(string& pinId)
{
	mFrameChecker = 0;
	return true;
}