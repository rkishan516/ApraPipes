#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "nvbuf_utils.h"
#include "EGL/egl.h"
#include "cudaEGL.h"
#include "CCDMA.h"
#include "CCDMACu.h"
#include "CCCuDMA.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "npp.h"
#include "CCKernel.h"
#include "RawImageMetadata.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"

#include <chrono>

using sys_clock = std::chrono::system_clock;

BOOST_AUTO_TEST_SUITE(ccdma_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(1920, 1080);
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	auto ccdma = boost::shared_ptr<Module>(new CCDMA(CCDMAProps(ImageMetadata::RGBA)));
	source->setNext(ccdma);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto ccdmacu = boost::shared_ptr<Module>(new CCDMACu(CCDMACuProps(ImageMetadata::RGB,stream)));
	ccdma->setNext(ccdmacu);

	auto cccudma = boost::shared_ptr<Module>(new CCCuDMA(CCCuDMAProps(ImageMetadata::RGBA,stream)));
	ccdmacu->setNext(cccudma);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	ccdma->setNext(sink);
	// auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(640, 0)));
	// cc->setNext(sink1);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(rgbatorgbkerneltest)
{
	void* input;
	cudaMalloc(&input, 1024*1024*4);
	cudaMemset(input,255,1024*1024*4);
	void* output;
	cudaMalloc(&output, 1024*1024*3*4);
	cudaMemset(output,0,1024*1024*4);
	
	cudaDeviceSynchronize();

	auto stream = cudastream_sp(new ApraCudaStream);
	auto cuStream = stream->getCudaStream();
	float avg_fps = 0;
    sys_clock::time_point current_time,end_time;
    for(int i=0;i<10;i++){
        current_time = sys_clock::now();
        for(int j=0;j<1000;j++){
			lanuchAPPRGBAToRGB(static_cast<Npp8u*>(input),1024*4,static_cast<Npp32f*>(output),1024*3, {1024,1024}, cuStream);
			cudaStreamSynchronize(cuStream);
        }
        end_time = sys_clock::now();
		std::chrono::nanoseconds diff = end_time - current_time;
		auto durationInSeconds =  diff.count()/ 1000000000.0;		
        std::cout << "Fps : " << 1000/(durationInSeconds) << std::endl; 
    }

	void* outputCpu;
	outputCpu = malloc(1024*1024*3*4);
	cudaMemcpy(outputCpu,output,1024*1024*3*4,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	float *verfier;
	verfier = static_cast<float*>(malloc(1024*1024*3*4));
	auto rgbSize = 1024*1024*3;
	for(auto i = 0; i < rgbSize; i++)
	{
		verfier[i] = 1;
	}
	
	int n = memcmp(verfier,outputCpu,1024*1024*3*4);
	LOG_ERROR << "..................." << n;
	// create input device pointer - cudamlloc - 1024x1024x4
	// memset random values - 255
	// create output device pointer - 1024x1024x3x4 
	// memset zero
	// two loops - outer loop is 10 and inner loop is 1000
	// start timer
	// call kernel in loop
	// end timer
	// verify output once at the last - 1024x1024x3x4 - 
	// create output cpu memory, cudamemcopy, 
	// create expected output cpu memory - loop through 1024x1024x3 - assign 255 
	// and std::memcompare
}

BOOST_AUTO_TEST_CASE(cvColorMap)
{
	void* input;
	cudaMalloc(&input, 1024*1024*4);
	cudaMemset(input,0.1,1024*1024*4);

	cv::Mat dest8u;
	cv::Mat im_color;
	cv::Mat dest(1024, 1024, CV_32FC1);

	float avg_fps = 0;
    sys_clock::time_point current_time,end_time;
    for(int i=0;i<10;i++){
        current_time = sys_clock::now();
        for(int j=0;j<1000;j++){
			cudaMemcpy(dest.data, input, 1024*1024*4, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			dest = 255*dest;
			dest.convertTo(dest8u, CV_8U);
			cv::applyColorMap(dest8u,im_color, cv::COLORMAP_JET);
        }
        end_time = sys_clock::now();
		std::chrono::nanoseconds diff = end_time - current_time;
		auto durationInSeconds =  diff.count()/ 1000000000.0;		
        std::cout << "Fps : " << 1000/(durationInSeconds) << std::endl; 
    }

	float *verfier;
	verfier = static_cast<float*>(malloc(1024*1024*3*4));
	auto rgbSize = 1024*1024*3;
	for(auto i = 0; i < rgbSize; i++)
	{
		verfier[i] = 1;
	}
	
	// int n = memcmp(verfier,im_color,1024*1024*3*4);
	// LOG_ERROR << "..................." << n;
}

BOOST_AUTO_TEST_CASE(cvColorConversion)
{
	void* input;
	cudaMalloc(&input, 1024*1024*4);
	cudaMemset(input,0.1,1024*1024*4);

	cv::Mat dest8u;
	cv::Mat im_color;
	cv::Mat dest(1024, 1024, CV_32FC1);

	float avg_fps = 0;
    sys_clock::time_point current_time,end_time;
    for(int i=0;i<10;i++){
        current_time = sys_clock::now();
        for(int j=0;j<1000;j++){
			cudaMemcpy(dest.data, input, 1024*1024*4, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			dest = dest/255;
			dest.convertTo(dest, CV_32FC1);
			cv::cvtColor(dest, dest,cv::COLOR_RGBA2RGB);
        }
        end_time = sys_clock::now();
		std::chrono::nanoseconds diff = end_time - current_time;
		auto durationInSeconds =  diff.count()/ 1000000000.0;		
        std::cout << "Fps : " << 1000/(durationInSeconds) << std::endl; 
    }

	float *verfier;
	verfier = static_cast<float*>(malloc(1024*1024*3*4));
	auto rgbSize = 1024*1024*3;
	for(auto i = 0; i < rgbSize; i++)
	{
		verfier[i] = 1;
	}
	
	// int n = memcmp(verfier,im_color,1024*1024*3*4);
	// LOG_ERROR << "..................." << n;
}

BOOST_AUTO_TEST_CASE(test)
{
	EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(eglDisplay == EGL_NO_DISPLAY)
	{
		throw AIPException(AIP_FATAL, "eglGetDisplay failed");
	} 
	
	if (!eglInitialize(eglDisplay, NULL, NULL))
	{
		throw AIPException(AIP_FATAL, "eglInitialize failed");
	} 
	DMAFDWrapper* dmafdWrapper = DMAFDWrapper::create(0,1024,1024,NvBufferColorFormat_ABGR32,NvBufferLayout_Pitch,eglDisplay);
	auto mapped = dmafdWrapper->hostPtr;
	memset(mapped,255,1024*1024*4);
	auto rgbSize = 10;
	for(auto i = 0; i < rgbSize; i++)
	{
		cout << (int)*(static_cast<uint8_t*>(mapped) + i) << " ";
	}
	cout <<endl;
}

BOOST_AUTO_TEST_CASE(fdtest)
{
	size_t size = 1024*1024*4;
	void *host = malloc(size);
	void *host_ref = malloc(size);

	cudaFree(0);

	EGLImageKHR eglInImage;
    CUgraphicsResource  pInResource;
    CUeglFrame eglInFrame;
	EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(eglDisplay == EGL_NO_DISPLAY)
	{
		throw AIPException(AIP_FATAL, "eglGetDisplay failed");
	} 
	
	if (!eglInitialize(eglDisplay, NULL, NULL))
	{
		throw AIPException(AIP_FATAL, "eglInitialize failed");
	} 
	DMAFDWrapper* dmafdWrapper = DMAFDWrapper::create(0,1024,1024,NvBufferColorFormat_ABGR32,NvBufferLayout_Pitch,eglDisplay);
	auto mapped = dmafdWrapper->hostPtr;
	
	
	auto src = DMAUtils::getCudaPtrForFD(dmafdWrapper->getFd(), eglInImage,&pInResource,eglInFrame, eglDisplay);

	int value = 128;
	for(int i = 0; i < 10; i++)
	{
		std::cout << "processing " << i << std::endl;
		value += i;
		memset(mapped, value, size);
		NvBufferMemSyncForDevice(dmafdWrapper->getFd(), 0, &mapped);
		cudaMemcpy(host, src, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		memset(host_ref, value, size);
		if(memcmp(host, host_ref, size) != 0)
		{
			std::cout << "failed" << std::endl;
		}

		value += 1;
		cudaMemset(src, value, size);
		cudaDeviceSynchronize();
		NvBufferMemSyncForCpu(dmafdWrapper->getFd(), 0, &mapped);
		memset(host_ref, value, size);
		if(memcmp(mapped, host_ref, size) != 0)
		{
			std::cout << "failed 2" << std::endl;
		}

		std::cout << "processing done " << i << std::endl;
	}
	


	DMAUtils::freeCudaPtr(eglInImage,&pInResource, eglDisplay);
}

BOOST_AUTO_TEST_SUITE_END()