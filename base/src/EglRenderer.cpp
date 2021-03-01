#include "Logger.h"
#include "EglRenderer.h"
#include "NvEglRenderer.h"
#include "DMAFDWrapper.h"

class EglRenderer::Detail
{

public:
	Detail(int _x_offset, int _y_offset): x_offset(_x_offset), y_offset(_y_offset) {}

	~Detail() 
    {
        if(renderer)
        {
            delete renderer;
        }
    }

    bool init(int height, int width){
        renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset, y_offset);
        // renderer1 = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset + width, y_offset);
        if (!renderer)
        {
            LOG_ERROR << "Failed to create EGL renderer";
            return false;
        }
        renderer->setFPS(60);

        return true;
    }

	bool shouldTriggerSOS()
	{
		return !renderer;
	}

	NvEglRenderer *renderer = nullptr;
    NvEglRenderer *renderer1 = nullptr;
    int x_offset,y_offset;
};

EglRenderer::EglRenderer(EglRendererProps props) : Module(SINK, "EglRenderer", props)
{
    mDetail.reset(new Detail(props.x_offset,props.y_offset));
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
    // mDetail->renderer1->render((static_cast<DMAFDWrapper *>(frame->data()))->getFd());
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
