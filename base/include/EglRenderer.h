#pragma once

#include "Module.h"

class EglRendererProps : public ModuleProps
{
public:
	EglRendererProps(int _x_offset,int _y_offset) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
	}
    int x_offset;
    int y_offset;
};

class EglRenderer : public Module
{
public:
    EglRenderer(EglRendererProps props);
    ~EglRenderer();

    bool init();
    bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool shouldTriggerSOS();
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};