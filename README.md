### To run TensorRT part:

```shell script
mkdir build
cd build
cmake ..
make -j8
./trt_sample -e ../sample.engine
```