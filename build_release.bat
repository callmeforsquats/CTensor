if not exist build_release mkdir build_release
cd build_releasee
cmake -G "Visual Studio 18 2026" -A x64 ..
cmake --build . --config Release
cd ..
pause