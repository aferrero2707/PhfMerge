cmake_minimum_required(VERSION 2.8.9)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(phfmerge CXX C)

# options controlling the build process
if(NOT INSTALL_PREFIX)
	set(INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
	MESSAGE( STATUS "Setting INSTALL_PREFIX from CMAKE_INSTALL_PREFIX" )
endif(NOT INSTALL_PREFIX)
MESSAGE( STATUS "INSTALL_PREFIX:         " ${INSTALL_PREFIX} )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_POSITION_INDEPENDENT_CODE True)

find_package(PkgConfig)

MESSAGE(STATUS "Check for LCMS2 availability...")
#FIND_PACKAGE(LCMS2)
pkg_check_modules(LCMS2 REQUIRED lcms2) 
MESSAGE( STATUS "LCMS2_INCLUDE_DIRS:         " ${LCMS2_INCLUDE_DIRS} )
MESSAGE( STATUS "LCMS2_LIBRARIES:            " ${LCMS2_LIBRARIES} )
MESSAGE( STATUS "LCMS2_LIBRARY_DIRS:         " ${LCMS2_LIBRARY_DIRS} )

find_package (TIFF REQUIRED)
MESSAGE( STATUS "TIFF_INCLUDE_DIRS:         " ${TIFF_INCLUDE_DIRS} )
MESSAGE( STATUS "TIFF_LIBRARIES:            " ${TIFF_LIBRARIES} )
MESSAGE( STATUS "TIFF_LIBRARY_DIRS:         " ${TIFF_LIBRARY_DIRS} )

find_package (PNG REQUIRED)

find_package (JPEG REQUIRED)
MESSAGE( STATUS "JPEG_INCLUDE_DIRS:         " ${JPEG_INCLUDE_DIRS} )
MESSAGE( STATUS "JPEG_LIBRARIES:            " ${JPEG_LIBRARIES} )
MESSAGE( STATUS "JPEG_LIBRARY_DIRS:         " ${JPEG_LIBRARY_DIRS} )

find_package (ZLIB REQUIRED)

pkg_check_modules(OPENEXR OpenEXR)
pkg_check_modules(ORC orc-0.4)

pkg_check_modules(VIPS REQUIRED vips>=8.5.1)
#pkg_check_modules(VIPSCC REQUIRED vipsCC)
MESSAGE( STATUS "VIPS_INCLUDE_DIRS:         " ${VIPS_INCLUDE_DIRS} )
MESSAGE( STATUS "VIPS_LIBRARY_DIRS:         " ${VIPS_LIBRARY_DIRS} )

  
IF(NOT MINGW)
#need to put here do setting LOCALE_DIR variable
if(USE_NLS)
find_package(Gettext)
if(Gettext_FOUND)
	if(NOT LOCALE_DIR)
    SET(LOCALE_DIR "${CMAKE_INSTALL_PREFIX}/share/locale")
	endif(NOT LOCALE_DIR)

	if(NOT LOCALE_DIR)
		message(QUIET "Set LOCALE_DIR to path to get Gettext working")
	else()
		find_package(Msgfmt)
		if(Msgfmt_FOUND)
			message(STATUS "Found msgfmt to convert language file. Translation enabled")
			add_subdirectory(po)
		else()
			message(STATUS "Cannot find msgfmt to convert language file. Translation won't be enabled")
		endif()
	endif(NOT LOCALE_DIR)
endif(Gettext_FOUND)
endif(USE_NLS)
ENDIF(NOT MINGW)
MESSAGE( STATUS "Msgfmt_FOUND:         " ${Msgfmt_FOUND} )




IF(MINGW)
  SET(COMPILE_FLAGS "-std=gnu++14 -march=nocona -mno-sse3 -mtune=generic -fno-ipa-sra -fpermissive")
ELSEIF(APPLE)
  SET(COMPILE_FLAGS "-march=nocona -mno-sse3 -mtune=generic -fpermissive")
ELSE(MINGW)
  SET(COMPILE_FLAGS "-std=gnu++14 -march=nocona -mno-sse3 -mtune=generic -Wno-error=narrowing -fno-ipa-sra -fpermissive")
ENDIF(MINGW)

set(COMPILE_FLAGS " ${COMPILE_FLAGS} -DINSTALL_PREFIX='\"${INSTALL_PREFIX}\"' ")
IF(APPLE)
  IF("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
	set(COMPILE_FLAGS "${COMPILE_FLAGS} -mmacosx-version-min=10.8 -stdlib=libc++ -std=c++14 -Wno-error=c++11-narrowing -Wno-c++11-extensions -Wno-c++14-extensions")
  ELSE("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
	set(COMPILE_FLAGS "${COMPILE_FLAGS} -mmacosx-version-min=10.8 -std=c++14")
  ENDIF("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
ENDIF()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-deprecated-register" COMPILER_SUPPORTS_W_NO_DEPRECATED_REGISTER)
IF(COMPILER_SUPPORTS_W_NO_DEPRECATED_REGISTER)
	set(COMPILE_FLAGS "${COMPILE_FLAGS} -Wno-deprecated-register")
ENDIF(COMPILER_SUPPORTS_W_NO_DEPRECATED_REGISTER)
CHECK_CXX_COMPILER_FLAG("-Wno-deprecated-declarations" COMPILER_SUPPORTS_W_NO_DEPRECATED_DECLARATIONS)
IF(COMPILER_SUPPORTS_W_NO_DEPRECATED_DECLARATIONS)
	set(COMPILE_FLAGS "${COMPILE_FLAGS} -Wno-deprecated-declarations")
ENDIF(COMPILER_SUPPORTS_W_NO_DEPRECATED_DECLARATIONS)

if(Msgfmt_FOUND)
	set(COMPILE_FLAGS "${COMPILE_FLAGS} -DLOCALEDIR=\"${LOCALE_DIR}\"")
endif()


# Check for openmp
find_package(OpenMP)
if(OpenMP_C_FLAGS AND USE_OPENMP)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} ${OpenMP_C_FLAGS}")
  set(COMPILE_FLAGS "${COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")
  if(WIN32)
    list(APPEND ADDITIONAL_LIBS gomp)
  endif(WIN32)
endif(OpenMP_C_FLAGS AND USE_OPENMP)


SET(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -DNDEBUG ${COMPILE_FLAGS} -Wno-unused-parameter") 
SET(CMAKE_CXX_FLAGS_RELEASE        " -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -DNDEBUG ${COMPILE_FLAGS}  -Wno-unused-parameter")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}")


set(WITH_SAN "address")
if(WITH_SAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -O1 -g -fsanitize=${WITH_SAN}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -O1 -g -fsanitize=${WITH_SAN}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O1 -g -fsanitize=${WITH_SAN}")
endif()





include_directories(${OPENEXR_INCLUDE_DIRS})
link_directories(${OPENEXR_LIBRARY_DIRS})
include_directories(${TIFF_INCLUDE_DIRS})
link_directories(${TIFF_LIBRARY_DIRS})
include_directories(${JPEG_INCLUDE_DIRS})
link_directories(${JPEG_LIBRARY_DIRS})
include_directories(${PNG_INCLUDE_DIRS})
link_directories(${PNG_LIBRARY_DIRS})

include_directories(${ORC_INCLUDE_DIRS})
link_directories(${ORC_LIBRARY_DIRS})

include_directories(${VIPS_INCLUDE_DIRS})
link_directories(${VIPS_LIBRARY_DIRS}  )

include_directories(${CMAKE_SOURCE_DIR}/src)

#FILE(GLOB Includes src/*.hh)
#FILE(GLOB Sources src/*.cc)



add_executable(phf_llf src/phf_llf.cc)
IF(APPLE)
  set_target_properties(phf_llf PROPERTIES LINK_FLAGS " -framework ApplicationServices ")
ENDIF(APPLE)

target_link_libraries(phf_llf
  ${TIFF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${LCMS2_LIBRARIES} 
  ${VIPS_LIBRARIES} ${VIPSCC_LIBRARIES}
  ${ZLIB_LIBRARIES}
  )

INSTALL(TARGETS phf_llf RUNTIME DESTINATION bin)
