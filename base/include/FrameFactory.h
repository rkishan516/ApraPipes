#pragma once
#include <boost/pool/object_pool.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <atomic>

#include "CommonDefs.h"
#include "FrameMetadata.h"
#include "Allocators.h"
#include <memory>

class FrameFactory
{
private:
	boost::object_pool<Frame> frame_allocator;
	std::shared_ptr<HostAllocator> memory_allocator; 
	
	frame_sp eosFrame;
	frame_sp emptyFrame;
	boost::mutex m_mutex;

	std::atomic_uint counter;
	std::atomic_size_t numberOfChunks;
	size_t maxConcurrentFrames;
public:
	FrameFactory(FrameMetadata::MemType memType, size_t _maxConcurrentFrames=0);
	virtual ~FrameFactory();
	frame_sp create(size_t size, boost::shared_ptr<FrameFactory>& mother);
	frame_sp create(frame_sp &frame, size_t size, boost::shared_ptr<FrameFactory>& mother);
	void destroy(Frame* pointer);
	frame_sp getEOSFrame() {
		return eosFrame;
	}

	frame_sp getEmptyFrame() { return emptyFrame; }

	std::string getPoolHealthRecord();
};
