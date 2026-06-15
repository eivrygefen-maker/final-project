cmake_minimum_required(VERSION 3.16)
project(stk_pgsm_guitar_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# STK discovery — STK_ROOT env, common paths, clear errors if missing
# ---------------------------------------------------------------------------
set(STK_ROOT "$ENV{STK_ROOT}" CACHE PATH "Root of STK install (include/ + lib/)")

if(NOT STK_ROOT OR STK_ROOT STREQUAL "")
    set(_stk_candidates
        "$ENV{HOME}/stk"
        "/home/vboxuser/stk"
        "/usr/local/stk"
        "/opt/stk"
        "${CMAKE_SOURCE_DIR}/../../third_party/stk"
    )
    foreach(_cand IN LISTS _stk_candidates)
        if(EXISTS "${_cand}/include/Stk.h")
            set(STK_ROOT "${_cand}")
            break()
        endif()
    endforeach()
endif()

set(STK_INCLUDE_DIR "${STK_ROOT}/include")
set(STK_LIB_DIR "${STK_ROOT}/build")

if(NOT EXISTS "${STK_INCLUDE_DIR}/Stk.h")
    message(FATAL_ERROR
        "STK headers not found.\n"
        "  Expected: ${STK_INCLUDE_DIR}/Stk.h\n"
        "  Set STK_ROOT to your STK install (contains include/Stk.h and build/libstk.*).\n"
        "  Example: export STK_ROOT=/home/vboxuser/stk"
    )
endif()

find_library(STK_LIBRARY
    NAMES stk libstk
    PATHS "${STK_LIB_DIR}" "${STK_ROOT}/lib" "/usr/local/lib"
    NO_DEFAULT_PATH
)
if(NOT STK_LIBRARY)
    find_library(STK_LIBRARY NAMES stk libstk)
endif()

if(NOT STK_LIBRARY)
    message(FATAL_ERROR
        "STK library not found (libstk).\n"
        "  Searched: ${STK_LIB_DIR}, ${STK_ROOT}/lib\n"
        "  Build STK first (make in STK source) or set STK_LIBRARY manually."
    )
endif()

message(STATUS "STK_ROOT=${STK_ROOT}")
message(STATUS "STK_INCLUDE_DIR=${STK_INCLUDE_DIR}")
message(STATUS "STK_LIBRARY=${STK_LIBRARY}")

add_executable(stk_pgsm_guitar_demo main.cpp)
target_include_directories(stk_pgsm_guitar_demo PRIVATE "${STK_INCLUDE_DIR}")
target_link_libraries(stk_pgsm_guitar_demo PRIVATE "${STK_LIBRARY}")

if(UNIX AND NOT APPLE)
    target_link_libraries(stk_pgsm_guitar_demo PRIVATE pthread)
endif()

# STK rawwaves (Plucked etc.) — optional runtime path hint
if(EXISTS "${STK_ROOT}/rawwaves")
    target_compile_definitions(stk_pgsm_guitar_demo PRIVATE STK_RAWWAVE_PATH="${STK_ROOT}/rawwaves")
endif()

install(TARGETS stk_pgsm_guitar_demo RUNTIME DESTINATION bin)
