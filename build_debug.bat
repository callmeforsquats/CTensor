if not exist build_debug mkdir build_debug
cd build_debug
cmake -G "Visual Studio 18 2026" -A x64 -DTREN_DEBUG=ON -DTREN_TESTS=ON ..
cmake --build . --config Debug
cd ..
echo Debug and Tests
pause