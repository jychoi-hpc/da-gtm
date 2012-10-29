include(../CMakeLists.comm.txt)

set(DAGTM_INTP_SRCS 
  intp.c
)

include_directories(${PROJECT_SOURCE_DIR}/comm)

add_executable(intp ${DAGTM_INTP_SRCS})

target_link_libraries(
  intp
  dagtmcomm
  ${HDF5_LIBRARIES}
  ${MPI_LIBRARIES}
)

#-----------------------------------------------------------------------------
# Option to enable USE_MKL
#-----------------------------------------------------------------------------
option (USE_MKL "Enable the use of MKL" ON)
if (USE_MKL)
  if (WIN32)
    file(TO_CMAKE_PATH $ENV{MKLROOT} MKL_HOME)
    set(MKL_INCLUDE_DIR "${MKL_HOME}/include")
    set(MKL_LIBRARY_DIR "${MKL_HOME}/lib/intel64")
  else (WIN32)
    set(MKL_INCLUDE_DIRS $ENV{MKLROOT}/include)
    set(MKL_LIBRARY_DIRS $ENV{MKLROOT}/lib)
  endif (WIN32)
  set(MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)
  link_directories(${MKL_LIBRARY_DIRS})
  include_directories(${MKL_INCLUDE_DIRS})
  
  ## Tip for WIN32 : To use MKL, don't use gslblas_dll
  target_link_libraries (
	intp
     ${GSL_GSL_LIBRARY}
	mkl_intel_lp64
	mkl_sequential
	mkl_core
  )
else (USE_MKL)
  target_link_libraries (
	intp
     ${GSL_GSL_LIBRARY}
     ${GSL_GSLCBLAS_LIBRARY}
  )
endif (USE_MKL)

