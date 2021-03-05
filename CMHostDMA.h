#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include <boost/pool/object_pool.hpp>


class CMHostDMAProps : public ModuleProps
{
public:
	CMHostDMAProps(ImageMetadata::ImageType _imageType)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
};

class CMHostDMA : public Module
{

public:
	CMHostDMA(CMHostDMAProps _props);
	virtual ~CMHostDMA();
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
	
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	CMHostDMAProps props;		
};