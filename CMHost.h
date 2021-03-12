#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include <boost/pool/object_pool.hpp>


class CMHostProps : public ModuleProps
{
public:
	CMHostProps(ImageMetadata::ImageType _imageType)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
};

class CMHost : public Module
{

public:
	CMHost(CMHostProps _props);
	virtual ~CMHost();
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
	CMHostProps props;		
};