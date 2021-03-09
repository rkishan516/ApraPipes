#include <boost/foreach.hpp>
#include "KeyStrokeModule.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include <ncurses.h>

class KeyStrokeModule::Detail
{
public:
	Detail() : mFrameSaved(0), mOpen(false),term(false)
	{
	}

	~Detail()
	{

	}

	void operator()()
	{
        while(!term){
			int k;
			k = getchar();
            if(k == 's'){
                mOpen = true;
                mFrameSaved = 0;
            }
        }
	}

    uint8_t mFrameSaved;
    bool mOpen;
	bool term;
	std::thread mThread;
};

KeyStrokeModule::KeyStrokeModule(KeyStrokeModuleProps _props) :Module(TRANSFORM, "KeyStrokeModule", _props), props(_props)
{
    mDetail.reset(new Detail());
	mDetail->mThread = std::thread(std::ref(*(mDetail.get())));
}

bool KeyStrokeModule::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool KeyStrokeModule::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

bool KeyStrokeModule::init()
{
	if (!Module::init())
	{
		return false;
	}
    initscr();

	return true;
}

bool KeyStrokeModule::term()
{
	mDetail->term = false;
	mDetail->mThread.join();
    endwin();
	return Module::term();
}

void KeyStrokeModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	addOutputPin(metadata, pinId);
}

bool KeyStrokeModule::process(frame_container& frames)
{
    if(mDetail->mOpen && (mDetail->mFrameSaved < props.nosFrame)){
        mDetail->mFrameSaved++;
        if(mDetail->mFrameSaved == props.nosFrame){
            mDetail->mOpen = false;
        }
        send(frames);
    }

	return true;
}