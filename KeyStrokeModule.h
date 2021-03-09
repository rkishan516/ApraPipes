#pragma once

#include "Module.h"
#include "CudaCommon.h"

class KeyStrokeModuleProps : public ModuleProps
{
public:
	KeyStrokeModuleProps(uint8_t _NosFrame) : ModuleProps() 
	{
        nosFrame = _NosFrame;
	}
    uint8_t nosFrame;
};

class KeyStrokeModule : public Module {
public:

	KeyStrokeModule(KeyStrokeModuleProps _props);
	virtual ~KeyStrokeModule() {}

	virtual bool init();
	virtual bool term();

protected:	
	bool process(frame_container& frames);
	bool validateInputPins();
	bool validateOutputPins();	
	void addInputPin(framemetadata_sp& metadata, string& pinId);

private:
    class Detail;
	boost::shared_ptr<Detail> mDetail;
	KeyStrokeModuleProps props;
};