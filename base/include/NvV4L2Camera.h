#pragma once

#include "Module.h"
#include "NvV4L2CameraHelper.h"

#include <memory>

class NvV4L2CameraProps : public ModuleProps
{
public:
	NvV4L2CameraProps(uint32_t _width, uint32_t _height) : ModuleProps(), width(_width), height(_height)
	{
		maxConcurrentFrames = 10;
	}

	uint32_t width;
	uint32_t height;
};

class NvV4L2Camera : public Module
{
public:
	NvV4L2Camera(NvV4L2CameraProps props);
	virtual ~NvV4L2Camera();
	bool init();
	bool term();

protected:
	bool produce();
	bool validateOutputPins();

private:
	uint32_t width;
	uint32_t height;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	std::shared_ptr<NvV4L2CameraHelper> mHelper;
};