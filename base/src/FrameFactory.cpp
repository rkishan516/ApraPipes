#include "stdafx.h"
#include <boost/bind.hpp>
#include "FrameFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"

#define LOG_FRAME_FACTORY

FrameFactory::FrameFactory(FrameMetadata::MemType memType, size_t _maxConcurrentFrames) : maxConcurrentFrames(_maxConcurrentFrames)
{
	switch (memType)
	{
		case FrameMetadata::MemType::HOST_PINNED:
			memory_allocator = std::make_shared<HostPinnedAllocator>();
			break;
		case FrameMetadata::MemType::CUDA_DEVICE:
			memory_allocator = std::make_shared<CudaDeviceAllocator>();
			break;
		case FrameMetadata::MemType::HOST:
			memory_allocator = std::make_shared<HostAllocator>();
			break;
		default:
			throw AIPException(AIP_FATAL, "Unknown MemType Requested<>" + std::to_string(memType));
			
	}
	eosFrame = frame_sp(new EoSFrame());
	emptyFrame = frame_sp(new EmptyFrame());
	counter = 0;
	numberOfChunks = 0;
}
FrameFactory::~FrameFactory()
{
}

size_t getNumberOfChunks(size_t size)
{
	return (size + APRA_CHUNK_SZ - 1) / APRA_CHUNK_SZ;
}

frame_sp FrameFactory::create(size_t size, boost::shared_ptr<FrameFactory> &mother)
{
	boost::mutex::scoped_lock lock(m_mutex);
	if (maxConcurrentFrames && counter >= maxConcurrentFrames)
	{
		return frame_sp();
	}
	size_t n = getNumberOfChunks(size);

	counter.fetch_add(1, memory_order_seq_cst);
	numberOfChunks.fetch_add(n, memory_order_seq_cst);

	return frame_sp(
		frame_allocator.construct(memory_allocator->allocateChunks(n), size, mother),
		boost::bind(&FrameFactory::destroy, this, _1));
}

void FrameFactory::destroy(Frame *pointer)
{
	boost::mutex::scoped_lock lock(m_mutex);
	counter.fetch_sub(1, memory_order_seq_cst);

	if (pointer->myOrig != NULL)
	{
		size_t n = getNumberOfChunks(pointer->size());
		numberOfChunks.fetch_sub(n, memory_order_seq_cst);
		memory_allocator->freeChunks(pointer->myOrig,n);
	}

	auto mother = pointer->myMother;
	pointer->~Frame();
	frame_allocator.free(pointer);
}

frame_sp FrameFactory::create(frame_sp &frame, size_t size, boost::shared_ptr<FrameFactory> &mother)
{
	size_t oldChunks = getNumberOfChunks(frame->size());
	size_t newChunks = getNumberOfChunks(size);
	size_t chunksToFree = oldChunks - newChunks;

	auto origPtr = frame->myOrig;
	if (origPtr == NULL)
	{
		throw AIPException(AIP_FATAL, string("oldFrame->myOrig in NULL. Not expected."));
	}

	if (chunksToFree < 0)
	{
		throw AIPException(AIP_NOTIMPLEMENTED, string("increasing chunks not yet implemented"));
	}

	boost::mutex::scoped_lock lock(m_mutex);
	counter.fetch_add(1, memory_order_seq_cst);

	if (chunksToFree > 0)
	{
		numberOfChunks.fetch_sub(chunksToFree, memory_order_seq_cst);
		auto ptr = (void *)((char *)origPtr + (newChunks * APRA_CHUNK_SZ));
		memory_allocator->freeChunks(ptr,chunksToFree);
	}

	frame->resetMemory(); // so that when destroyBuffer is called it should not free the memory
	return frame_sp(
		frame_allocator.construct(origPtr, size, mother),
		boost::bind(&FrameFactory::destroy, this, _1));
}

std::string FrameFactory::getPoolHealthRecord()
{
	std::ostringstream stream;
	stream << "Chunks<" << numberOfChunks << "> TotalBytes<" << numberOfChunks * APRA_CHUNK_SZ << "> Frames<" << counter << ">";

	return stream.str();
}
