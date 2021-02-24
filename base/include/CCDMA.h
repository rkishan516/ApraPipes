#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include "ExtFrame.h"
#include "DMABuffer.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class CCDMAProps : public ModuleProps
{
public:
	CCDMAProps(ImageMetadata::ImageType _imageType)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
};

class CCDMA : public Module
{

public:
	CCDMA(CCDMAProps _props);
	virtual ~CCDMA();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId); // throws exception if validation fails
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};