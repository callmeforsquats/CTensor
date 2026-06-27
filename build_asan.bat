
if not exist build_asan mkdir build_asan
cd build_asan
cmake -G "Visual Studio 18 2026" -A x64 -DTREN_DEBUG=ON -DTREN_ASAN=ON -DTREN_TESTS=ON ..
cmake --build . --config Debug
cd ..
echo Running tests with AddressSanitizer...
pause