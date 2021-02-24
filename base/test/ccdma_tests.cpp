#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "NvV4L2Camera.h"
#include "EglRenderer.h"
#include "CCDMA.h"

BOOST_AUTO_TEST_SUITE(ccdma_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	NvV4L2CameraProps sourceProps(640, 480);
	sourceProps.fps = 60;
	auto source = boost::shared_ptr<Module>(new NvV4L2Camera(sourceProps));

	auto cc = boost::shared_ptr<Module>(new CCDMA(CCDMAProps(ImageMetadata::RGBA)));
	source->setNext(cc);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	cc->setNext(sink);
	auto sink1 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(640, 0)));
	cc->setNext(sink1);

	PipeLine p("test");
	p.appendModule(source);
	BOOST_TEST(p.init());

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10000));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()