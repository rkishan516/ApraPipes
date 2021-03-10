#include <iostream>
#include <ctime>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "CCDMA.h"
#include "CCDMACu.h"
#include "TensorRT.h"
#include "CCCuDMA.h"
#include "FileWriterModule.h"
#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "CudaMemCopy.h"
#include "QuePushStrategy.h"
#include "FramesMuxer.h"
#include "CudaStreamSynchronize.h"
#include "CCDMAHost.h"
#include "CMHostDMA.h"
#include "TestModule.h"
#include "GLTransform.h"
#include "CuCtxSynchronize.h"
#include "KeyStrokeModule.h"
#include "CCSaver.h"

typedef struct
{
    const char * file;
    const char* outputDirectory;
    unsigned int save_n_frame;
    bool enable_display;
} context_t;

static void
print_usage(void)
{
    std::cout << ("\n\tUsage: trt_sample [OPTIONS]\n\n"
           "\tExample: \n"
           "\t./trt_sample -e ../sample.engine -n 30 -o ./out/ \n\n"
           "\tSupported options:\n"
           "\t-e\t\tSet TensorRT engine file\n"
           "\t-n\t\tSave the next n frames after Pressing s\n"
           "\t-o\t\tOutput directory where frames to be stored\n"
           "\t-d\t\tdisable display\n"
           "\t-h\t\tPrint this usage\n\n"
           "\tNOTE: It runs infinitely until you terminate it with <ctrl+c>\n");
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->file = "../sample.engine";
    ctx->outputDirectory = "./out/";
    ctx->save_n_frame = 0;
    ctx->enable_display = true;
}

static bool
parse_cmdline(context_t * ctx, int argc, char **argv)
{
    int c;

    if (argc < 2)
    {
        print_usage();
        exit(EXIT_SUCCESS);
    }
    ctx->save_n_frame = 0;
    while ((c = getopt(argc, argv, "e:n:o:dh")) != -1)
    {
        switch (c)
        {
            case 'e':
                ctx->file = optarg;
                break;
            case 'n':
                ctx->save_n_frame = strtol(optarg, NULL, 10);
                break;
            case 'd':
                ctx->enable_display = false;
                break;
            case 'o':
                ctx->outputDirectory = optarg;
                break;
            case 'h':
                print_usage();
                exit(EXIT_SUCCESS);
                break;
            default:
                print_usage();
                return false;
        }
    }

    return true;
}


void newPipeLine(context_t *ctx){
    cudaFree(0);
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.qlen = 1;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);
    
    auto stream = cudastream_sp(new ApraCudaStream);
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.qlen = 1;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    TensorRTProps tensorrtprops(ctx->file,stream);
    tensorrtprops.qlen = 1;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    ccdmacu->setNext(tensorrt);

    FramesMuxerProps muxerProps;
    muxerProps.maxDelay = 200;
    muxerProps.qlen = 1;
    auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
    tensorrt->setNext(muxer);
    ccdma->setNext(muxer);

    CCCuDMAProps cccudmaprops(ImageMetadata::RGBA, stream);
    cccudmaprops.qlen = 1;
    auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(cccudmaprops));
	muxer->setNext(cccudma);

    CuCtxSynchronizeProps cuCtxSyncProps;
    cuCtxSyncProps.qlen = 1;
    auto cuctx = boost::shared_ptr<Module>(new CuCtxSynchronize(cuCtxSyncProps));
	cccudma->setNext(cuctx);

    GLTransformProps gltransformProps(ImageMetadata::RGBA, 512, 1024, 0 ,0);
    gltransformProps.qlen = 1;
    auto gltransform = boost::shared_ptr<Module>(new GLTransform(gltransformProps));
	cuctx->setNext(gltransform);

    EglRendererProps eglProps(0, 0,512, 1024);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
    eglProps.qlen = 1;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	gltransform->setNext(sink);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(259200));
	p.stop();
	p.term();
	p.wait_for_all();
}

void keyStrokePipeLine(context_t *ctx){
    cudaFree(0);

    /* Common Pipe */
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.qlen = 1;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    auto stream = cudastream_sp(new ApraCudaStream);

    /* KeyStroke Pipe */
    KeyStrokeModuleProps keystrokeProps(ctx->save_n_frame);
    keystrokeProps.qlen = 1;
    auto keystroke = boost::shared_ptr<Module>(new KeyStrokeModule(keystrokeProps));
    ccdma->setNext(keystroke);

    CCSaverProps ccsaverprops(ImageMetadata::RGB, stream);
    ccsaverprops.qlen = 1;
    auto ccsaver = boost::shared_ptr<Module>(new CCSaver(ccsaverprops));
    keystroke->setNext(ccsaver);

    auto copyProps = CudaMemCopyProps(cudaMemcpyDeviceToHost, stream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	ccsaver->setNext(copy);

    CuCtxSynchronizeProps cuCtxSyncProps0;
    cuCtxSyncProps0.qlen = 1;
    auto cuctx0 = boost::shared_ptr<Module>(new CuCtxSynchronize(cuCtxSyncProps0));
	copy->setNext(cuctx0);

    std::string path( std::string(ctx->outputDirectory) + "???.raw" );
    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(path.c_str(), true)));
	cuctx0->setNext(fileWriter);

    
    /* Renderer Pipe */
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.qlen = 1;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    TensorRTProps tensorrtprops(ctx->file,stream);
    tensorrtprops.qlen = 1;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    ccdmacu->setNext(tensorrt);

    FramesMuxerProps muxerProps;
    muxerProps.maxDelay = 200;
    muxerProps.qlen = 1;
    auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
    tensorrt->setNext(muxer);
    ccdma->setNext(muxer);

    CCCuDMAProps cccudmaprops(ImageMetadata::RGBA, stream);
    cccudmaprops.qlen = 1;
    auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(cccudmaprops));
	muxer->setNext(cccudma);

    CuCtxSynchronizeProps cuCtxSyncProps;
    cuCtxSyncProps.qlen = 1;
    auto cuctx = boost::shared_ptr<Module>(new CuCtxSynchronize(cuCtxSyncProps));
	cccudma->setNext(cuctx);

    GLTransformProps gltransformProps(ImageMetadata::RGBA, 1024, 512, 0 ,0);
    gltransformProps.qlen = 1;
    if(!ctx->enable_display){
        gltransformProps.logHealth = true;
    	gltransformProps.logHealthFrequency = 100;
    }
    auto gltransform = boost::shared_ptr<Module>(new GLTransform(gltransformProps));
	cuctx->setNext(gltransform);
    if(ctx->enable_display){
        EglRendererProps eglProps(0, 0);
        eglProps.logHealth = true;
        eglProps.logHealthFrequency = 100;
        eglProps.qlen = 1;
        auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
        gltransform->setNext(sink);
    }

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(259200));
	p.stop();
	p.term();
	p.wait_for_all();
}

// main pipeline ------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    context_t ctx;
    set_defaults(&ctx);
    if(parse_cmdline(&ctx, argc, argv)){
        // TensorRTTest();
        // LOG_ERROR << "Starting Null Test..................";
        // nullTest();
        // LOG_ERROR << "Starting SGL Test..................";
        // sgl();
        // LOG_ERROR << "Starting SGLR1 Test..................";
        // sglr1();
        // LOG_ERROR << "Starting SGLR1R2 Test..................";
        // sglr1r2();
        // LOG_ERROR << "Starting SGLCCCS Test..................";
        // sglcccs();
        // LOG_ERROR << "Starting SGLCCCPUCS Test..................";
        // sglccCpucs();
        // LOG_ERROR << "Starting SGLCCTRTCS Test..................";
        // sglcctrtcs();
        // LOG_ERROR << "Starting SGLCCTRTCMCS Test..................";
        // sglcctrtcmcs();
        // LOG_ERROR << "Starting SGLCCCPUTRTCSCMCPU Test..................";
        // sglccCputrtcscmCpu();
        // LOG_ERROR << "Starting SGLCCCPUTRTCSCMR1R2 Test..................";
        // sglccCputrtcscmr1r2();
        // LOG_ERROR << "Starting Full Pipeline Test..................";
        // pipelineFunction();
        // LOG_ERROR << "Starting FRHDCMCSR1 Pipeline Test..................";
        // frhdcmcsr1();
        // fileReaderTest();
        // LOG_ERROR << "Starting Full New Pipeline Test..................";
        // newPipeLine(&ctx);
        LOG_ERROR << "Starting Full New KeyStroke Pipeline Test..................";
        keyStrokePipeLine(&ctx);
    }
    return 0;
}
