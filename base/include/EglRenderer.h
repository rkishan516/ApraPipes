#pragma once

#include "Module.h"

class EglRendererProps : public ModuleProps
{
public:
	EglRendererProps(int _x_offset,int _y_offset, int _height, int _width) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
		height = _height;
		width = _width;
	}
	EglRendererProps(int _x_offset,int _y_offset) : ModuleProps()
	{
        x_offset = _x_offset;
        y_offset = _y_offset;
	}
    int x_offset;
    int y_offset;
	int height;
	int width;
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