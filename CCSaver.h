#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include "ExtFrame.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class CCSaverProps : public ModuleProps
{
public:
	CCSaverProps(ImageMetadata::ImageType _imageType)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
};

class CCSaver : public Module
{

public:
	CCSaver(CCSaverProps _props);
	virtual ~CCSaver();
	bool init();
	bool term();

protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool validateOutputPins();
	void addInputPin(framemetadata_sp& metadata, string& pinId);		
	bool shouldTriggerSOS();
	bool processEOS(string& pinId);

private:
	void setMetadata(framemetadata_sp& metadata);
	class Detail;
	boost::shared_ptr<Detail> mDetail;

	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	CCSaverProps props;		
};