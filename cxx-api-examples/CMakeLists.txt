include_directories(${PROJECT_SOURCE_DIR})

add_executable(streaming-zipformer-cxx-api ./streaming-zipformer-cxx-api.cc)
target_link_libraries(streaming-zipformer-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-zipformer-with-hr-cxx-api ./streaming-zipformer-with-hr-cxx-api.cc)
target_link_libraries(streaming-zipformer-with-hr-cxx-api sherpa-onnx-cxx-api)

add_executable(speech-enhancement-gtcrn-cxx-api ./speech-enhancement-gtcrn-cxx-api.cc)
target_link_libraries(speech-enhancement-gtcrn-cxx-api sherpa-onnx-cxx-api)

add_executable(kws-cxx-api ./kws-cxx-api.cc)
target_link_libraries(kws-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-zipformer-rtf-cxx-api ./streaming-zipformer-rtf-cxx-api.cc)
target_link_libraries(streaming-zipformer-rtf-cxx-api sherpa-onnx-cxx-api)

add_executable(whisper-cxx-api ./whisper-cxx-api.cc)
target_link_libraries(whisper-cxx-api sherpa-onnx-cxx-api)

add_executable(fire-red-asr-cxx-api ./fire-red-asr-cxx-api.cc)
target_link_libraries(fire-red-asr-cxx-api sherpa-onnx-cxx-api)

add_executable(moonshine-cxx-api ./moonshine-cxx-api.cc)
target_link_libraries(moonshine-cxx-api sherpa-onnx-cxx-api)

add_executable(sense-voice-cxx-api ./sense-voice-cxx-api.cc)
target_link_libraries(sense-voice-cxx-api sherpa-onnx-cxx-api)

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  add_executable(sense-voice-simulate-streaming-microphone-cxx-api
    ./sense-voice-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(sense-voice-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(parakeet-tdt-simulate-streaming-microphone-cxx-api
    ./parakeet-tdt-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(parakeet-tdt-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )
endif()

if(SHERPA_ONNX_HAS_ALSA)
  add_executable(sense-voice-simulate-streaming-alsa-cxx-api
    ./sense-voice-simulate-streaming-alsa-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/alsa.cc
  )
  target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api asound)
  endif()
endif()

add_executable(sense-voice-with-hr-cxx-api ./sense-voice-with-hr-cxx-api.cc)
target_link_libraries(sense-voice-with-hr-cxx-api sherpa-onnx-cxx-api)

add_executable(dolphin-ctc-cxx-api ./dolphin-ctc-cxx-api.cc)
target_link_libraries(dolphin-ctc-cxx-api sherpa-onnx-cxx-api)

add_executable(vad-cxx-api ./vad-cxx-api.cc)
target_link_libraries(vad-cxx-api sherpa-onnx-cxx-api)

if(SHERPA_ONNX_ENABLE_TTS)
  add_executable(matcha-tts-zh-cxx-api ./matcha-tts-zh-cxx-api.cc)
  target_link_libraries(matcha-tts-zh-cxx-api sherpa-onnx-cxx-api)

  add_executable(matcha-tts-en-cxx-api ./matcha-tts-en-cxx-api.cc)
  target_link_libraries(matcha-tts-en-cxx-api sherpa-onnx-cxx-api)

  add_executable(kokoro-tts-en-cxx-api ./kokoro-tts-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-en-cxx-api sherpa-onnx-cxx-api)

  add_executable(kokoro-tts-zh-en-cxx-api ./kokoro-tts-zh-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-zh-en-cxx-api sherpa-onnx-cxx-api)
endif()

# =================================================================
# 创建流式语音识别包装库 - 完全静态链接版本
# =================================================================

if(TARGET sherpa-onnx-cxx-api)
  add_library(sense_voice_streaming_wrapper SHARED
    sense-voice-streaming-wrapper.cc
  )

  target_compile_features(sense_voice_streaming_wrapper PRIVATE cxx_std_14)
  target_compile_options(sense_voice_streaming_wrapper PRIVATE 
    -fPIC
    -fvisibility=hidden
    -fvisibility-inlines-hidden
  )

  # 获取sherpa-onnx-cxx-api的所有依赖库
  get_target_property(SHERPA_CXX_LINK_LIBS sherpa-onnx-cxx-api INTERFACE_LINK_LIBRARIES)
  get_target_property(SHERPA_CXX_LINK_LIBS_PRIVATE sherpa-onnx-cxx-api LINK_LIBRARIES)
  
  # 强制静态链接所有sherpa-onnx相关库
  target_link_libraries(sense_voice_streaming_wrapper
    -Wl,--whole-archive
    sherpa-onnx-cxx-api
    sherpa-onnx-c-api
    sherpa-onnx-core
    -Wl,--no-whole-archive
  )

  # 查找并静态链接ONNX Runtime
  find_library(ONNXRUNTIME_LIB 
    NAMES libonnxruntime.a onnxruntime
    PATHS 
      ${CMAKE_BINARY_DIR}/_deps/onnxruntime-src/lib
      ${CMAKE_BINARY_DIR}/lib
      /usr/local/lib
      /usr/lib
    NO_DEFAULT_PATH
  )
  
  if(ONNXRUNTIME_LIB)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
    if(ONNXRUNTIME_LIB MATCHES "\\.a$")
      # 静态库
      target_link_libraries(sense_voice_streaming_wrapper ${ONNXRUNTIME_LIB})
    else()
      # 动态库，尝试静态链接
      target_link_libraries(sense_voice_streaming_wrapper
        -Wl,--whole-archive
        ${ONNXRUNTIME_LIB}
        -Wl,--no-whole-archive
      )
    endif()
  else()
    message(WARNING "ONNX Runtime library not found, trying target-based linking")
    if(TARGET onnxruntime)
      target_link_libraries(sense_voice_streaming_wrapper
        -Wl,--whole-archive
        onnxruntime
        -Wl,--no-whole-archive
      )
    endif()
  endif()

  # 静态链接标准库和其他依赖
  target_link_libraries(sense_voice_streaming_wrapper
    -static-libgcc
    -static-libstdc++
    -Wl,--as-needed
    -Wl,--exclude-libs,ALL
  )

  # 只保留必要的系统库
  target_link_libraries(sense_voice_streaming_wrapper
    -lpthread
    -ldl
    -lm
  )

  # 设置更严格的链接选项
  set_target_properties(sense_voice_streaming_wrapper PROPERTIES
    VERSION 1.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map -Wl,--no-undefined"
  )

  target_include_directories(sense_voice_streaming_wrapper PRIVATE
    ${CMAKE_SOURCE_DIR}
  )
  
  # 添加编译时定义来避免某些动态库依赖
  target_compile_definitions(sense_voice_streaming_wrapper PRIVATE
    SHERPA_ONNX_STATIC_LINK=1
  )
  
  message(STATUS "Building sense_voice_streaming_wrapper with complete static linking")
  
  # 添加自定义命令来检查依赖
  add_custom_command(TARGET sense_voice_streaming_wrapper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Checking dependencies of libsense_voice_streaming_wrapper.so:"
    COMMAND ldd $<TARGET_FILE:sense_voice_streaming_wrapper> || echo "ldd check completed"
    COMMAND ${CMAKE_COMMAND} -E echo "Checking file size:"
    COMMAND ls -lh $<TARGET_FILE:sense_voice_streaming_wrapper>
    COMMAND ${CMAKE_COMMAND} -E echo "Checking undefined symbols:"
    COMMAND nm -D $<TARGET_FILE:sense_voice_streaming_wrapper> | grep -E "^[[:space:]]*U " || echo "No undefined symbols found"
    COMMENT "Verifying static linking"
  )
  
else()
  message(WARNING "sherpa-onnx-cxx-api not available, skipping sense_voice_streaming_wrapper")
endif()
