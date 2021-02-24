#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class CCDMACuProps : public ModuleProps
{
public:
	CCDMACuProps(ImageMetadata::ImageType _imageType)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
};

class CCDMACu : public Module
{

public:
	CCDMACu(CCDMACuProps _props);
	virtual ~CCDMACu();
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
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	CCDMACuProps props;		
};