#!/usr/bin/env python3

import subprocess
import sys
import os

def check_library_dependencies(lib_path):
    """检查动态库的依赖"""
    if not os.path.exists(lib_path):
        print(f"Library not found: {lib_path}")
        return False
    
    print(f"Checking dependencies for: {lib_path}")
    print("=" * 60)
    
    # 检查文件大小（静态链接的库通常更大）
    file_size = os.path.getsize(lib_path) / (1024 * 1024)  # MB
    print(f"Library size: {file_size:.2f} MB")
    
    # 检查ldd输出
    try:
        result = subprocess.run(['ldd', lib_path], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        system_libs = {'linux-vdso.so', 'ld-linux-x86-64.so', '/lib64/ld-linux-x86-64.so.2'}
        essential_libs = {'libpthread.so', 'libdl.so', 'libm.so', 'libc.so', 'libgcc_s.so', 'libstdc++.so'}
        # 移除librt.so，因为它不是必需的
        
        problematic_deps = []
        all_deps = []
        
        for line in lines:
            if '=>' in line:
                lib_name = line.split('=>')[0].strip()
                lib_path_full = line.split('=>')[1].strip().split()[0] if len(line.split('=>')[1].strip().split()) > 0 else ''
                
                all_deps.append((lib_name, lib_path_full))
                
                # 检查是否是系统库或基础库
                is_system = any(sys_lib in lib_name for sys_lib in system_libs)
                is_essential = any(ess_lib in lib_name for ess_lib in essential_libs)
                
                # 特别检查sherpa-onnx和onnxruntime相关库
                is_sherpa = 'sherpa-onnx' in lib_name or 'onnxruntime' in lib_name
                
                if not is_system and not is_essential:
                    problematic_deps.append((lib_name, lib_path_full, is_sherpa))
                
                print(f"  {lib_name} => {lib_path_full}")
            else:
                print(f"  {line}")
        
        print(f"\nTotal dependencies: {len(all_deps)}")
        print(f"Library size: {file_size:.2f} MB")
        
        print("\n" + "=" * 60)
        if problematic_deps:
            print("⚠️  NON-ESSENTIAL DEPENDENCIES FOUND:")
            sherpa_deps = []
            other_deps = []
            
            for lib_name, lib_path, is_sherpa in problematic_deps:
                if is_sherpa:
                    sherpa_deps.append((lib_name, lib_path))
                else:
                    other_deps.append((lib_name, lib_path))
            
            if sherpa_deps:
                print("\n🔴 CRITICAL - Sherpa-ONNX/ONNX Runtime dependencies (should be static):")
                for lib_name, lib_path in sherpa_deps:
                    print(f"  - {lib_name} => {lib_path}")
            
            if other_deps:
                print("\n🟡 OTHER dependencies:")
                for lib_name, lib_path in other_deps:
                    print(f"  - {lib_name} => {lib_path}")
            
            # 如果只有非关键依赖，可能是可接受的
            if not sherpa_deps and file_size > 50:  # 大文件可能意味着静态链接成功
                print("\n✅ ACCEPTABLE: Only non-critical dependencies found, and library size suggests static linking")
                return True
            else:
                print(f"\n❌ PROBLEMATIC: Found {len(sherpa_deps)} critical dependencies")
                return False
        else:
            print("✅ ALL DEPENDENCIES ARE ACCEPTABLE")
            print("Only system and essential libraries are linked dynamically.")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"Error running ldd: {e}")
        return False
    except FileNotFoundError:
        print("ldd command not found")
        return False

def check_symbols(lib_path):
    """检查导出的符号"""
    print(f"\nChecking exported symbols for: {lib_path}")
    print("=" * 60)
    
    try:
        result = subprocess.run(['nm', '-D', lib_path], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        exported_symbols = []
        for line in lines:
            if line.strip() and not line.startswith(' ') and ' T ' in line:
                symbol = line.split()[-1]
                if not symbol.startswith('_') or symbol.startswith('_Z'):  # C symbols or mangled C++
                    exported_symbols.append(symbol)
        
        print("Exported symbols:")
        for symbol in sorted(exported_symbols):
            print(f"  {symbol}")
            
        return len(exported_symbols) > 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error running nm: {e}")
        return False

if __name__ == "__main__":
    lib_paths = [
        "/root/sherpa-onnx/build/lib/libsense_voice_streaming_wrapper.so",
        "/root/sherpa-onnx/build/cxx-api-examples/libsense_voice_streaming_wrapper.so",
    ]
    
    success = False
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            deps_ok = check_library_dependencies(lib_path)
            symbols_ok = check_symbols(lib_path)
            success = deps_ok and symbols_ok
            break
    
    if not success:
        print("\n❌ Library verification failed!")
        print("\nTry running the build script:")
        print("chmod +x ./build_static_sensevoice_streaming_wrapper.sh")
        print("./build_static_sensevoice_streaming_wrapper.sh")
        sys.exit(1)
    else:
        print("\n✅ Library verification passed!")
        print("The library is ready for Python usage with minimal dependencies.")
