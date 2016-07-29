include (../CMakeLists.comm.txt)

##add_definitions (-Wno-write-strings)

set (DAGTM_SRCS 
    dagtm.cxx
    ##message.pb.cc
    )

include_directories(${PROJECT_SOURCE_DIR}/comm)

#-----------------------------------------------------------------------------
# Option to enable PVIZRPC_SERVER_ENABLE
#-----------------------------------------------------------------------------
option (PVIZRPC_SERVER_ENABLE "Enable PvizRPC Server" OFF)
if (PVIZRPC_SERVER_ENABLE)
set (DAGTM_SRCS 
    ${DAGTM_SRCS}
    message.pb.cc
    )
  add_definitions(-DDAGTM_PVIZRPC_SERVER_ON)

  ## Protobuf
  find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h $ENV{PROTOBUF_HOME}/include)
  find_library(Protobuf_LIBRARIES NAMES protobuf PATHS $ENV{PROTOBUF_HOME}/lib)
  include_directories(${Protobuf_INCLUDE_DIR})
  
  ## Protorpc
  find_path(Protorpc_INCLUDE_DIR protorpc.pb.h $ENV{PROTORPC_HOME}/include)
  find_library(Protorpc_LIBRARIES NAMES protorpc PATHS $ENV{PROTORPC_HOME}/lib)
  include_directories(${Protorpc_INCLUDE_DIR})

  ## Boost
  find_package(Boost 1.44 REQUIRED system)
  include_directories (${Boost_INCLUDE_DIRS})

  ## PvizRPC
  set(Pvizrpc_BASENAME pvizrpc)
  set(Pvizrpc_OUTFILES ${Pvizrpc_BASENAME}.pb.h ${Pvizrpc_BASENAME}.pb.cc)
  
  add_custom_command(
    OUTPUT ${Pvizrpc_OUTFILES}
    COMMAND protoc ${Pvizrpc_BASENAME}.proto --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    ${Pvizrpc_BASENAME}.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  include_directories (${CMAKE_CURRENT_BINARY_DIR})
  list(APPEND DAGTM_SRCS ${Pvizrpc_OUTFILES})

endif(PVIZRPC_SERVER_ENABLE)

#-----------------------------------------------------------------------------
# Option to enable NB_SERVER_ENABLE
#-----------------------------------------------------------------------------
option (NB_SERVER_ENABLE "Enable NB Server" OFF)
if (NB_SERVER_ENABLE)
  add_definitions(-DDAGTM_NB_SERVER_ON)
  include_directories(${PROJECT_SOURCE_DIR}/nbclient/include)
endif()

#-----------------------------------------------------------------------------
# Option to enable ACTIVEMQ_SERVER_ENABLE
#-----------------------------------------------------------------------------
option (ACTIVEMQ_SERVER_ENABLE "Enable ActiveMQ Server" OFF)
if (ACTIVEMQ_SERVER_ENABLE)
  add_definitions(-DDAGTM_ACTIVEMQ_SERVER_ON)
  add_definitions(-DDARWIN)
  add_definitions(-DSIGPROCMASK_SETS_THREAD_MASK)

  ## Activemq-CPP
  find_path(Activemqcpp_INCLUDE_DIR activemq/library/ActiveMQCPP.h $ENV{ACTIVEMQCPP_HOME}/include/activemq-cpp-3.4.0)
  find_library(Activemqcpp_LIBRARIES NAMES activemq-cpp PATHS $ENV{ACTIVEMQCPP_HOME}/lib)

  message(ACTIVEMQCPP_HOME = $ENV{ACTIVEMQCPP_HOME})
  message(Activemqcpp_INCLUDE_DIR = ${Activemqcpp_INCLUDE_DIR})
  message(Activemqcpp_LIBRARIES = ${Activemqcpp_LIBRARIES})
  include_directories(${Activemqcpp_INCLUDE_DIR})

  ## APR
  find_library(Apr_LIBRARIES NAMES apr-1 PATHS $ENV{APR_HOME}/lib)
  message(Apr_LIBRARIES = ${Apr_LIBRARIES})

  ## APR-UTIL
  find_library(Aprutil_LIBRARIES NAMES aprutil-1 PATHS $ENV{APRUTIL_HOME}/lib)
  message(Aprutil_LIBRARIES = ${Aprutil_LIBRARIES})
  
  ## Protobuf
  find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h $ENV{PROTOBUF_HOME}/include)
  find_library(Protobuf_LIBRARIES NAMES protobuf PATHS $ENV{PROTOBUF_HOME}/lib)
  include_directories(${Protobuf_INCLUDE_DIR})
endif()

add_executable (dagtm ${DAGTM_SRCS})

target_link_libraries (
  dagtm
  dagtmcomm
  ${HDF5_LIBRARIES}
  ${MPI_LIBRARIES}
  -lz
)

if (WIN32)
target_link_libraries (
  dagtm
  Ws2_32.lib
)
endif (WIN32)

if (PVIZRPC_SERVER_ENABLE)
  target_link_libraries (dagtm ${Protobuf_LIBRARIES})
  target_link_libraries (dagtm ${Protorpc_LIBRARIES})
  target_link_libraries (dagtm ${Boost_LIBRARIES})
endif(PVIZRPC_SERVER_ENABLE)

if (ACTIVEMQ_SERVER_ENABLE)
  target_link_libraries (dagtm 
    ${Activemqcpp_LIBRARIES}
    ${Protobuf_LIBRARIES}
    /opt/local/lib/libapr-1.dylib
    /opt/local/lib/libaprutil-1.dylib
    /opt/local/lib/db46/libdb-4.6.dylib
    /opt/local/lib/libexpat.dylib
    /opt/local/lib/libiconv.dylib
    -lssl
    -lcrypto
    -lpthread
  )
endif(ACTIVEMQ_SERVER_ENABLE)

#-----------------------------------------------------------------------------
# Option to enable USE_MKL
#-----------------------------------------------------------------------------
#option (USE_MKL "Enable the use of MKL" ON)
if (USE_MKL AND (NOT USE_ACML))
  #if (WIN32)
  #  file(TO_CMAKE_PATH $ENV{MKLROOT} MKL_HOME)
  #  set(MKL_INCLUDE_DIR "${MKL_HOME}/include")
  #  set(MKL_LIBRARY_DIR "${MKL_HOME}/lib/intel64")
  #else (WIN32)
  #  set(MKL_INCLUDE_DIRS $ENV{MKLROOT}/include)
  #  set(MKL_LIBRARY_DIRS $ENV{MKLROOT}/lib)
  #endif (WIN32)
   
  #set(MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)
  #link_directories(${MKL_LIBRARY_DIRS})
  #var_message(MKL_LIBRARY_DIRS)
  include_directories(${MKL_INCLUDE_DIR})
  
  ## Tip for WIN32 : To use MKL, don't use gslblas_dll
  target_link_libraries (
    dagtm
    ${GSL_GSL_LIBRARY}
    ${MKL_LIBRARIES}
  )
endif ()

#-----------------------------------------------------------------------------
# Option to enable USE_ACML
#-----------------------------------------------------------------------------
if (USE_ACML AND (NOT USE_MKL))
  target_link_libraries (
    dagtm
    ${GSL_GSL_LIBRARY}
    ${ACML_CBLAS_LIBRARY}
  )
endif ()

if ((NOT USE_MKL) AND (NOT USE_ACML))
  target_link_libraries (
    dagtm
    ${GSL_GSL_LIBRARY}
    ${GSL_GSLCBLAS_LIBRARY}
  )
endif ()


var_message(GSL_GSL_LIBRARY)

#-----------------------------------------------------------------------------
# Option to enable USE_IPP
#-----------------------------------------------------------------------------
#option (USE_IPP "Enable the use of IPP" OFF)
if (USE_IPP)
  if (WIN32)
  elseif (APPLE)
    #include_directories ($ENV(IPPROOT)/include)
    target_link_libraries (
      dagtm
      $ENV{IPPROOT}/lib/libippcore.dylib
      $ENV{IPPROOT}/lib/libipps.dylib
    )
  else ()
    find_path(IPP_INCLUDE_DIR
      NAMES ipp.h
      HINTS
        ENV IPP_HOME IPPROOT
      PATH_SUFFIXES
        include Include
    )
    include_directories(IPP_INCLUDE_DIR)
    target_link_libraries (
      dagtm
      $ENV{IPPROOT}/lib/intel64/libippcore.so
      $ENV{IPPROOT}/lib/intel64/libipps.so
    )
  endif ()

endif (USE_IPP)

