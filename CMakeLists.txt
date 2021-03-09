cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
SET( CONFIG_FILEPATH ${CMAKE_SOURCE_DIR}/config.cmake)
SET ( JETSON_MULTIMEDIA_LIB_INCLUDE "/usr/src/jetson_multimedia_api/include" )	
include("./Hunter.cmake")
HunterGate(
	URL "https://github.com/saichaitanyaeruvaram/hunter/archive/opencv-4.5.0.tar.gz"
    SHA1 "1258e48a1c06ce027b6d3aa141c7f9661ff0d843"
	FILEPATH ${CONFIG_FILEPATH}
  )

OPTION(ENABLE_LINUX "Use this switch to enable LINUX" ON)
OPTION(ENABLE_CUDA "Use this switch to enable CUDA" ON)
OPTION(ENABLE_ARM64 "Use this switch to enable ARM64" ON)

IF(ENABLE_CUDA)
  add_compile_definitions(APRA_CUDA_ENABLED)
ENDIF(ENABLE_CUDA)

IF(ENABLE_LINUX)
  add_compile_definitions(LINUX)	
ENDIF(ENABLE_LINUX)

IF(ENABLE_ARM64)
  add_compile_definitions(ARM64)	
ENDIF(ENABLE_ARM64)

project(TensorRTSample)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
hunter_add_package(OpenCV)
find_package(OpenCV)
hunter_add_package(Boost COMPONENTS system thread filesystem serialization log chrono)
find_package(Boost CONFIG REQUIRED system thread filesystem serialization log chrono)
list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra)
list(APPEND CMAKE_PREFIX_PATH /usr/lib/aarch64-linux-gnu/tegra-egl)
find_package(Curses REQUIRED)
enable_language(CUDA)
find_library(V4L2LIB v4l2)
find_library(EGLLIB EGL)
find_library(GLESv2LIB GLESv2)
find_library(X11LIB X11)
find_library(NVBUFUTILSLIB nvbuf_utils PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(APRAPIPES_LIB libaprapipes.a PATHS /mnt/disks/ssd/kishan/ApraPipes/_build/ NO_DEFAULT_PATH)
SET(JETSON_LIBS
			libcudart_static.a	
			libcuda.so.1.1
			${V4L2LIB}
			${NVBUFUTILSLIB}
			${EGLLIB}	
			${GLESv2LIB}
			${X11LIB}
)
SET(CORE
	TensorRT.h
	TensorRT.cpp
	CCDMAHost.h
	CCDMAHost.cpp
	CMHostDMA.h
	CMHostDMA.cpp
	TestModule.h
	TestModule.cpp
	launchDivision.cu
	GLTransform.cpp
	KeyStrokeModule.h
	KeyStrokeModule.cpp
	CCSaver.h
	CCSaver.cpp
)

message("CORE -------------  " ${CORE})
# TARGETS --------------------------------------------------------------------------------------------------------------
include_directories(AFTER SYSTEM include
    ${BOOST_ROOT}/include
    ${OPENCV_ROOT}/include/opencv4
  	/usr/local/cuda-10.2/include
    /mnt/disks/ssd/kishan/ApraPipes/base/include
	${JETSON_MULTIMEDIA_LIB_INCLUDE} 
	${CURSES_INCLUDE_DIR}
)

add_library(trt_samples STATIC ${CORE})

add_executable(trt_sample trt_sample.cpp )

target_include_directories(trt_sample PUBLIC
	${JETSON_MULTIMEDIA_LIB_INCLUDE} 
	${CUDA_INCLUDE_DIRS} 
	${TensorRT_INCLUDE_DIRS} 
)

target_link_libraries(trt_sample PUBLIC 
	trt_samples
	${APRAPIPES_LIB}
	Boost::log
	Boost::filesystem
	Boost::serialization	
	Boost::thread  
	Boost::chrono
	Boost::system
	Boost::boost
	${OpenCV_LIBS}
	${JETSON_LIBS}
	${CUDA_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${TensorRT_LIBRARIES}
	${CURSES_LIBRARIES}
)
