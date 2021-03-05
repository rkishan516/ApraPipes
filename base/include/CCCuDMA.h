#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class CCCuDMAProps : public ModuleProps
{
public:
	CCCuDMAProps(ImageMetadata::ImageType _imageType, cudastream_sp& _stream)
	{
		stream_sp = _stream;
		stream = _stream->getCudaStream();
		imageType = _imageType;
	}
	cudastream_sp stream_sp;
	cudaStream_t stream;
	ImageMetadata::ImageType imageType;	
};

class CCCuDMA : public Module
{

public:
	CCCuDMA(CCCuDMAProps _props);
	virtual ~CCCuDMA();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;

	bool mNoChange;
	int mInputFrameType;
	int mOutputFrameType;
	size_t mFrameChecker;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	CCCuDMAProps props;		
};