#include "Logger.h"
#include "EglRenderer.h"
#include "NvEglRenderer.h"
#include "DMAFDWrapper.h"

class EglRenderer::Detail
{

public:
	Detail(int _x_offset, int _y_offset, int _width, int _height): x_offset(_x_offset), y_offset(_y_offset), width(_width), height(_height) {}

	~Detail() 
    {
        if(renderer)
        {
            delete renderer;
        }
    }

    bool init(int _height, int _width){
        uint32_t displayHeight, displayWidth;
        NvEglRenderer::getDisplayResolution(displayWidth,displayHeight);
        if(height && width){
            x_offset += (displayWidth-width)/2;
            y_offset += (displayHeight-height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset, y_offset);
        }else{
            x_offset += (displayWidth-_width)/2;
            y_offset += (displayHeight-_height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, _width, _height, x_offset, y_offset);
        }
        if (!renderer)
        {
            LOG_ERROR << "Failed to create EGL renderer";
            return false;
        }
        // #Mar10_Feedback - is it required ? can you remove it ?
        renderer->setFPS(30);

        return true;
    }

	bool shouldTriggerSOS()
	{
		return !renderer;
	}

	NvEglRenderer *renderer = nullptr;
    // #Mar10_Feedback - change to uint32_t
    int x_offset,y_offset,width,height;
};

EglRenderer::EglRenderer(EglRendererProps props) : Module(SINK, "EglRenderer", props)
{
    mDetail.reset(new Detail(props.x_offset,props.y_offset, props.width, props.height));
}

EglRenderer::~EglRenderer() {}

bool EglRenderer::init(){
    if (!Module::init())
	{
		return false;
	}
    return true;
}

bool EglRenderer::process(frame_container& frames)
{
    auto frame = frames.cbegin()->second;
	if (isFrameEmpty(frame))
	{
		return true;
	}
    
    mDetail->renderer->render((static_cast<DMAFDWrapper *>(frame->data()))->getFd());
    return true;
}

bool EglRenderer::validateInputPins(){
    if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

    framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

    return true;
}

bool EglRenderer::term(){
    bool res = Module::term();
    return res;
}

bool EglRenderer::processSOS(frame_sp& frame)
{
	auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(frame->getMetadata());
    mDetail->init(metadata->getHeight(),metadata->getWidth());
	return true;
}

bool EglRenderer::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
} 
