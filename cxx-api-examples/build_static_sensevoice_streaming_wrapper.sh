#!/bin/bash

set -e

echo "Building sherpa-onnx with static linking..."

# 进入项目根目录
cd /root/sherpa-onnx

# 清理之前的构建
rm -rf build
mkdir -p build
cd build

# 配置CMake with 静态链接偏好
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
      -DSHERPA_ONNX_ENABLE_TESTS=OFF \
      -DSHERPA_ONNX_ENABLE_CHECK=OFF \
      -DCMAKE_FIND_LIBRARY_SUFFIXES=".a;.so" \
      -DCMAKE_CXX_FLAGS="-fPIC -pthread" \
      -DCMAKE_C_FLAGS="-fPIC -pthread" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++ -pthread" \
      -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -pthread" \
      ..

echo "Building sherpa-onnx libraries..."
make -j$(nproc) sherpa-onnx-cxx-api sherpa-onnx-c-api sherpa-onnx-core

echo "Building sense_voice_streaming_wrapper..."
make -j$(nproc) sense_voice_streaming_wrapper VERBOSE=1

echo "Checking the built library..."
if [ -f "lib/libsense_voice_streaming_wrapper.so" ]; then
    echo "✅ Library built successfully: lib/libsense_voice_streaming_wrapper.so"
    ls -lh lib/libsense_voice_streaming_wrapper.so
elif [ -f "cxx-api-examples/libsense_voice_streaming_wrapper.so" ]; then
    echo "✅ Library built successfully: cxx-api-examples/libsense_voice_streaming_wrapper.so"
    ls -lh cxx-api-examples/libsense_voice_streaming_wrapper.so
else
    echo "❌ Library not found!"
    exit 1
fi

echo "Running dependency check..."
python /root/sherpa-onnx/cxx-api-examples/check_dependencies.py
