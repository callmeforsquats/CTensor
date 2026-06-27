# # cmake_minimum_required(VERSION 3.18)
# # project(Tren LANGUAGES C CUDA)

# # # ============= ОПЦИИ =============
# # option(TREN_BUILD_TESTS "Build unit tests" OFF)
# # option(TREN_DEBUG "Enable debug mode" OFF)
# # option(TREN_PROFILE "Enable profiling" OFF)
# # option(TREN_ASAN "Enable AddressSanitizer" OFF)

# # find_package(CUDAToolkit REQUIRED)

# # # Настройки для компиляции "вслепую"
# # set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
# # set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
# # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # # Автоматический поиск всех .cpp и .cu файлов в папке src
# # file(GLOB_RECURSE LIB_SOURCES "src/*.c" "src/*.cu")


# # # Создание статической библиотеки из всех найденных файлов
# # add_library(tren STATIC ${LIB_SOURCES})


# # if (TREN_BUILD_TESTS)
# #     file(GLOB_RECURSE TEST_SOURCES "tests/*.c")
# #     add_executable(unit_test ${TEST_SOURCES})
# #     # --- Настройка путей для Теста (в корень проекта) ---

# #     set_target_properties(unit_test PROPERTIES 
# #         RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"

# #         # Также обнуляем суффиксы конфигураций для MSVC
# #         RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
# #         OUTPUT_NAME "test"
# #     )
# #     target_link_libraries(unit_test PRIVATE tren)
# # endif()

# # if(MSVC)
# #     # Для обычных .c файлов
# #     target_compile_options(tren PRIVATE $<$<COMPILE_LANGUAGE:C>:/TC>)
# #     # Для хост-части .cu файлов (передаем флаг внутрь nvcc)
# #     target_compile_options(tren PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /TC>)
    
# #     target_compile_options(unit_test PRIVATE /TC)
# # endif()

# # # ============= САНИТАЙЗЕР =============
# # if(TREN_ASAN)
# #     if(MSVC)
# #         add_compile_options(/fsanitize=address /Zi /Od /MTd /fno-omit-frame-pointer)
# #         add_link_options(/fsanitize=address /DEBUG /INCREMENTAL:NO)
# #     else()
# #         add_compile_options(-fsanitize=address -g -O0 -fno-omit-frame-pointer)
# #         add_link_options(-fsanitize=address)
# #     endif()
# #     message(STATUS "⚠️ AddressSanitizer is ENABLED (performance may be slower)")
# # endif()


# # # Настройка CUDA (обязательно для работы ядер из библиотеки)
# # set_target_properties(tren PROPERTIES 
# #     CUDA_SEPARABLE_COMPILATION ON
# #     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# #     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist"
# #     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist"
# #     # Для Windows/MSVC нужно обнулить суффиксы конфигураций (Release/Debug)
# #     ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/dist"
# #     LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/dist"
# # )




# # # Прокидываем заголовки и рантайм


# # target_include_directories(tren PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# # target_include_directories(tren PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# # target_link_libraries(tren PUBLIC CUDA::cudart)





























# #---------CMAKE VERSION------------
# cmake_minimum_required(VERSION 3.18)
# project(Tren LANGUAGES C CUDA)

# #-----------OPTIONS--------------
# option(TREN_TESTS "Build tests" OFF)
# option(TREN_DEBUG "Debug mode" OFF)
# option(TREN_ASAN "AddressSanitizer" OFF)


# #--------FIND PACKAGES-------------
# find_package(CUDAToolkit REQUIRED)
# find_package(OpenMP REQUIRED)      
# set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
 

# #------COLLECTING FILES-----------------------
# file(GLOB_RECURSE SRC "src/*.c" "src/*.cu")
# add_library(tren STATIC ${SRC})

# #-------COMMON COMPILER FLAGS (C & CUDA)
# target_compile_options(tren PRIVATE
#     $<$<OR:$<C_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:/W3>
#     $<$<AND:$<NOT:$<C_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:C>>:-Wall -Wextra>

#     $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:MSVC>>:/openmp>
#     $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<C_COMPILER_ID:MSVC>>>:-fopenmp>

#     $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<C_COMPILER_ID:MSVC>>:-Xcompiler /openmp>
#     $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<C_COMPILER_ID:MSVC>>>:-Xcompiler -fopenmp>
# )

# if(TREN_DEBUG)
# #----------DEBUG OPTIONS-----------------
#     target_compile_options(tren PRIVATE 
#         $<$<OR:$<C_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:/Od /Zi>
#         $<$<AND:$<NOT:$<C_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:C>>:-g -O0>
#     )
# else()
# #---------RELEASE OPTIONS-------------------
#     target_compile_options(tren PRIVATE 
#         $<$<OR:$<C_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:/O2>
#         $<$<AND:$<NOT:$<C_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:C>>:-O2>
#     )
# endif()


# #----------MEMORY SANITIZER------------------
# if(TREN_ASAN)
#     if(MSVC)
#         target_compile_options(tren PUBLIC /fsanitize=address /Zi)
#         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /INCREMENTAL:NO")
#     else()
#         target_compile_options(tren PUBLIC -fsanitize=address -g)
#         target_link_options(tren PUBLIC -fsanitize=address)
#     endif()
# endif()


# #-----------INCLUDE DIRECTORIES---------------
# target_include_directories(tren PUBLIC include PRIVATE src)

# #------------LIBRARIES LINKS---------------------
# target_link_libraries(tren PRIVATE CUDA::cudart OpenMP::OpenMP_C)

# #-----------COMPILED .lib PROPS--------------------
# set_target_properties(tren PROPERTIES
#     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist"
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )

# #---------TESTS OPTIONS---------------------------------
# if(TREN_TESTS)
#     file(GLOB_RECURSE TEST_SRC "tests/*.c")
#     add_executable(test ${TEST_SRC})
#     target_link_libraries(test tren)
#     target_include_directories(test PRIVATE src include)
#     set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist")
# endif()



cmake_minimum_required(VERSION 3.18)
project(Tren LANGUAGES C CUDA)

# На Windows помогаем IntelliSense заранее, чтобы не было красного кода
if(WIN32)
    add_definitions(-D_OPENMP=202011)
endif()

#-----------OPTIONS--------------
option(TREN_TESTS "Build tests" OFF)
option(TREN_DEBUG "Debug mode" OFF)
option(TREN_ASAN "AddressSanitizer" OFF)

#--------FIND PACKAGES-------------
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------COLLECTING FILES-----------------------
file(GLOB_RECURSE SRC "src/*.c" "src/*.cu")
add_library(tren STATIC ${SRC})

#-------COMPILER FLAGS (Разделяем логику)---------

# 1. Базовые ворнинги
target_compile_options(tren PRIVATE
    $<$<COMPILE_LANGUAGE:C>:
        $<$<OR:$<C_COMPILER_ID:MSVC>,$<C_COMPILER_ID:Clang>>:/W3>
        $<$<C_COMPILER_ID:GNU>:-Wall -Wextra>
    >
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /W3>
)

# 2. Оптимизации (Главная часть для Clang)
if(TREN_DEBUG)
    target_compile_options(tren PRIVATE 
        $<$<COMPILE_LANGUAGE:C>:
            $<$<OR:$<C_COMPILER_ID:MSVC>,$<C_COMPILER_ID:Clang>>:/Od /Zi /RTC1>
            $<$<C_COMPILER_ID:GNU>:-g -O0>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:/Od /Zi -Xcompiler /RTC1>
    )
else()
    target_compile_options(tren PRIVATE 
        $<$<COMPILE_LANGUAGE:C>:
            # Если работаем через Clang-cl (Windows)
            $<$<C_COMPILER_ID:Clang>:
                /clang:-O3
                /clang:-ffast-math
                /clang:-mavx2
                /clang:-mfma
                /clang:-march=native
            >
            $<$<C_COMPILER_ID:MSVC>:/O2 /arch:AVX2 /fp:fast>
            $<$<C_COMPILER_ID:GNU>:-O3 -mavx2 -mfma -ffast-math -march=native>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler /O2 -Xcompiler /arch:AVX2>
    )
endif()

# 3. Современный OpenMP (LLVM vs MSVC)
if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(tren PRIVATE $<$<COMPILE_LANGUAGE:C>:/openmp /clang:-fopenmp>)
    target_link_libraries(tren PRIVATE libomp)
    # Явно указываем путь к omp.h из Build Tools, если он не подхватится сам
    # target_include_directories(tren PRIVATE "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/Llvm/lib/clang/20/include")
    
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(tren PRIVATE OpenMP::OpenMP_C)
endif()

#----------MEMORY SANITIZER------------------
if(TREN_ASAN)
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(tren PUBLIC /fsanitize=address /Zi)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /INCREMENTAL:NO")
    else()
        target_compile_options(tren PUBLIC -fsanitize=address -g)
        target_link_options(tren PUBLIC -fsanitize=address)
    endif()
endif()

#-----------INCLUDE & LIBRARIES---------------
target_include_directories(tren PUBLIC include PRIVATE src)
target_link_libraries(tren PRIVATE CUDA::cudart)

#-----------PROPERTIES--------------------
set_target_properties(tren PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/dist/Debug"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/dist/Release"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#---------TESTS---------------------------------
if(TREN_TESTS)
    file(GLOB_RECURSE TEST_SRC "tests/*.c")
    add_executable(test ${TEST_SRC})
    target_link_libraries(test tren)
    target_include_directories(test PRIVATE src include)
    set_target_properties(test PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/dist/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/dist/Release"
    )
endif()
