#pragma once

#include "Module.h"
#include "CudaCommon.h"

#include "ExtFrame.h"
#include "DMABuffer.h"
#include <boost/pool/object_pool.hpp>

#include <deque>

class GLTransformProps : public ModuleProps
{
public:
	GLTransformProps(ImageMetadata::ImageType _imageType, int _width, int _height,  int _top , int _left) : top(_top) , left(_left) , width(_width) , height(_height)
	{
		imageType = _imageType;
	}
	ImageMetadata::ImageType imageType;	
	int top,left,width,height;
};

class GLTransform : public Module
{

public:
	GLTransform(GLTransformProps _props);
	virtual ~GLTransform();
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