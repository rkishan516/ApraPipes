#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "CCDMA.h"
#include "CCDMACu.h"
#include "CCCuDMA.h"

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

BOOST_AUTO_TEST_SUITE_END()