include(../CMakeLists.comm.txt)

set(DAGTMLIB_SRCS 
  vec.c vec.h
  comm.c comm.h
  getopt.c getopt.h
  h5comm.c h5comm.h
  hilbert.c hilbert.h
  log.c log.h
  mpicomm.c mpicomm.h
)

if (WIN32)
elseif (APPLE)
  execute_process(
    COMMAND sysctl -n hw.cpufrequency
    COMMAND tr -d "\n"
    RESULT_VARIABLE CMD_OUT
    OUTPUT_VARIABLE CPU_HZ
  )

  if (CPU_HZ GREATER 0)
    execute_process(
      COMMAND echo ${CPU_HZ} / 1000000
      COMMAND bc
      COMMAND tr -d "\n"
      RESULT_VARIABLE CMD_OUT
      OUTPUT_VARIABLE CPU_MHZ
    )

    message(${CPU_MHZ} " MHz CPU detected.")
    add_definitions(-DCPU_MHZ=${CPU_MHZ})
  else ()
    message("CPU Frequency detection failure")
  endif ()
else ()
  if (NOT CPU_MHZ)
    execute_process(
      COMMAND dmesg 
      COMMAND sed -n "s/.*Detected \\([0-9\\.]*\\) MHz processor./\\1/p"
      COMMAND tr -d "\n"
      RESULT_VARIABLE CMD_OUT
      OUTPUT_VARIABLE CPU_MHZ
    )

    if (CPU_MHZ GREATER 0)
      message(${CPU_MHZ} " MHz CPU detected.")
      add_definitions(-DCPU_MHZ=${CPU_MHZ})
    else ()
      message("CPU frequency detection failure")
    endif ()
  else ()
    add_definitions(-DCPU_MHZ=${CPU_MHZ})
  endif()
endif ()

add_library(dagtmcomm ${DAGTMLIB_SRCS})


