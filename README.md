# 1. 进入构建目录
cd /root/sherpa-onnx
mkdir -p build
cd build

# 2. 配置CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
      -DSHERPA_ONNX_ENABLE_TESTS=OFF \
      -DSHERPA_ONNX_ENABLE_CHECK=OFF \
      ..

# 3. 编译主库
make -j$(nproc) sherpa-onnx-cxx-api

# 4. 编译包装库
make -j$(nproc) sense_voice_streaming_wrapper

# 5. 检查生成的库文件
ls -la lib/libsense_voice_streaming_wrapper.so*
ls -la cxx-api-examples/libsense_voice_streaming_wrapper.so*