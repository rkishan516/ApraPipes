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

void nullTest(){
    PipeLine p("test");
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sgl(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglr1(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA,960,270,0,0);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    EglRendererProps eglProps(0, 0,960,270);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
    ccdma->setNext(sink);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglr1r2(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    EglRendererProps eglProps(0, 0,1280, 640);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
    ccdma->setNext(sink);

	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(eglProps));
    ccdma->setNext(sink1);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglcccs(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    auto stream = cudastream_sp(new ApraCudaStream);
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.logHealth = true;
	ccdmacuprops.logHealthFrequency = 100;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    CudaStreamSynchronizeProps streamProps(stream);
    streamProps.logHealth = true;
	streamProps.logHealthFrequency = 100;
    auto cs = boost::shared_ptr<Module>(new CudaStreamSynchronize(streamProps));
    ccdmacu->setNext(cs);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglccCpucs(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    CCDMAHostProps ccdmahostprops(ImageMetadata::RGB);
    ccdmahostprops.logHealth = true;
	ccdmahostprops.logHealthFrequency = 100;
	auto ccdmahost = boost::shared_ptr<Module>(new CCDMAHost(ccdmahostprops));
	ccdma->setNext(ccdmahost);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	ccdmahost->setNext(copy);

    auto copyProps1 = CudaMemCopyProps(cudaMemcpyDeviceToHost, stream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(copyProps1));
	copy->setNext(copy1);


    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglcctrtcs(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    auto stream = cudastream_sp(new ApraCudaStream);
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.logHealth = true;
	ccdmacuprops.logHealthFrequency = 100;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    ccdmacu->setNext(tensorrt);

    CudaStreamSynchronizeProps streamProps(stream);
    streamProps.logHealth = true;
	streamProps.logHealthFrequency = 100;
    auto cs = boost::shared_ptr<Module>(new CudaStreamSynchronize(streamProps));
    tensorrt->setNext(cs);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglcctrtcmcs(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    auto stream = cudastream_sp(new ApraCudaStream);
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.logHealth = true;
	ccdmacuprops.logHealthFrequency = 100;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    ccdmacu->setNext(tensorrt);

    CCCuDMAProps cccudmaprops(ImageMetadata::RGBA,stream);
    cccudmaprops.logHealth = true;
	cccudmaprops.logHealthFrequency = 100;
    auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(cccudmaprops));
	tensorrt->setNext(cccudma);

    CudaStreamSynchronizeProps streamProps(stream);
    streamProps.logHealth = true;
	streamProps.logHealthFrequency = 100;
    auto cs = boost::shared_ptr<Module>(new CudaStreamSynchronize(streamProps));
    cccudma->setNext(cs);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglccCputrtcscmCpu(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    CCDMAHostProps ccdmahostprops(ImageMetadata::RGB);
    ccdmahostprops.logHealth = true;
	ccdmahostprops.logHealthFrequency = 100;
	auto ccdmahost = boost::shared_ptr<Module>(new CCDMAHost(ccdmahostprops));
	ccdma->setNext(ccdmahost);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	ccdmahost->setNext(copy);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    copy->setNext(tensorrt);

    auto copyProps1 = CudaMemCopyProps(cudaMemcpyDeviceToHost, stream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(copyProps1));
	tensorrt->setNext(copy1);

    CudaStreamSynchronizeProps streamProps(stream);
    streamProps.logHealth = true;
	streamProps.logHealthFrequency = 100;
    auto cs = boost::shared_ptr<Module>(new CudaStreamSynchronize(streamProps));
    copy1->setNext(cs);

    CMHostDMAProps cmhostdmaprops(ImageMetadata::RGBA);
    cmhostdmaprops.logHealth = true;
	cmhostdmaprops.logHealthFrequency = 100;
    auto cmhostdma = boost::shared_ptr<Module>(new CMHostDMA(cmhostdmaprops));
	cs->setNext(cmhostdma);

    auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
    ccdma->setNext(muxer);
    cmhostdma->setNext(muxer);
    auto muxerOutputs = muxer->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE);
    std::vector<std::string> leftPin = {muxerOutputs[1]};
    std::vector<std::string> rightPin = {muxerOutputs[0]};

    EglRendererProps eglProps(0, 0,1280, 640);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	muxer->setNext(sink, leftPin);
	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(640, 0,1280, 640)));
	muxer->setNext(sink1, rightPin);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}

void sglccCputrtcscmr1r2(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);

    CCDMAHostProps ccdmahostprops(ImageMetadata::RGB);
    ccdmahostprops.logHealth = true;
	ccdmahostprops.logHealthFrequency = 100;
	auto ccdmahost = boost::shared_ptr<Module>(new CCDMAHost(ccdmahostprops));
	ccdma->setNext(ccdmahost);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	ccdmahost->setNext(copy);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    copy->setNext(tensorrt);

    CCCuDMAProps cccudmaprops(ImageMetadata::RGBA,stream);
    cccudmaprops.logHealth = true;
	cccudmaprops.logHealthFrequency = 100;
    auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(cccudmaprops));
	tensorrt->setNext(cccudma);

    CudaStreamSynchronizeProps streamProps(stream);
    streamProps.logHealth = true;
	streamProps.logHealthFrequency = 100;
    auto cs = boost::shared_ptr<Module>(new CudaStreamSynchronize(streamProps));
    cccudma->setNext(cs);
    
    auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
    ccdma->setNext(muxer);
    cs->setNext(muxer);
    auto muxerOutputs = muxer->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE);
    std::vector<std::string> leftPin = {muxerOutputs[1]};
    std::vector<std::string> rightPin = {muxerOutputs[0]};

    EglRendererProps eglProps(0, 0,540, 540);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	muxer->setNext(sink, leftPin);
	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(540, 0,540, 540)));
	muxer->setNext(sink1, rightPin);

    PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
}


void pipelineFunction(){
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    NvV4L2CameraProps sourceProps(1920, 1080, 10);
	sourceProps.fps = 60;
    sourceProps.logHealth = true;
	sourceProps.logHealthFrequency = 100;
    sourceProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

    CCDMAProps ccdmaprops(ImageMetadata::RGBA);
    ccdmaprops.logHealth = true;
	ccdmaprops.logHealthFrequency = 100;
	auto ccdma = boost::shared_ptr<Module>(new CCDMA(ccdmaprops));
	source->setNext(ccdma);
    
    auto stream = cudastream_sp(new ApraCudaStream);
    CCDMACuProps ccdmacuprops(ImageMetadata::RGB,stream);
    ccdmacuprops.logHealth = true;
	ccdmacuprops.logHealthFrequency = 100;
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(ccdmacuprops));
	ccdma->setNext(ccdmacu);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    ccdmacu->setNext(tensorrt);

    CCCuDMAProps cccudmaprops(ImageMetadata::RGBA, stream);
    cccudmaprops.logHealth = true;
	cccudmaprops.logHealthFrequency = 100;
    auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(cccudmaprops));
	tensorrt->setNext(cccudma);
    
    auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
    ccdma->setNext(muxer);
    cccudma->setNext(muxer);
    auto muxerOutputs = muxer->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE);
    std::vector<std::string> leftPin = {muxerOutputs[1]};
    std::vector<std::string> rightPin = {muxerOutputs[0]};

    EglRendererProps eglProps(0, 0,540, 540);
    eglProps.logHealth = true;
	eglProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new EglRenderer(eglProps));
	muxer->setNext(sink, leftPin);
	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(540, 0,540, 540)));
	muxer->setNext(sink1, rightPin);

	PipeLine p("test");
	p.appendModule(source);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(600));
	p.stop();
	p.term();
	p.wait_for_all();
}

void fileReaderTest(){
    auto width = 1024;
	auto height = 1024;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./../test_rgb.raw")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::RGB, CV_8UC3, width*3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata);

    auto stream = cudastream_sp(new ApraCudaStream);
    auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

    auto testModule = boost::shared_ptr<Module>(new Test(TestProps(stream,true)));
    copy->setNext(testModule);

    TensorRTProps tensorrtprops("../sample.engine",stream);
    tensorrtprops.logHealth = true;
	tensorrtprops.logHealthFrequency = 100;
    tensorrtprops.qlen = 1;
    auto tensorrt = boost::shared_ptr<Module>(new TensorRT(tensorrtprops));
    testModule->setNext(tensorrt);

    auto testModule1 = boost::shared_ptr<Module>(new Test(TestProps(stream,false)));
    tensorrt->setNext(testModule1);

    auto copyProps1 = CudaMemCopyProps(cudaMemcpyDeviceToHost, stream);
	auto copy1 = boost::shared_ptr<Module>(new CudaMemCopy(copyProps1));
	testModule1->setNext(copy1);

    auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("../test_mono.raw", true)));
	copy1->setNext(fileWriter);

	fileReader->init();
	copy->init();
    testModule->init();
    tensorrt->init();
    testModule1->init();
    copy1->init();
	fileWriter->init();

	fileReader->play(true);


	for (auto i = 0; i < 1; i++)
	{
		fileReader->step();
        copy->step();
        testModule->step();
        tensorrt->step();
        testModule1->step();
        copy1->step();
        fileWriter->step();
	}

}