#!/bin/bash
usage ()
{
    echo "USAGE : `basename $0` [options] <compiler = [gnu | intel | cray]>"
    echo "Options"
    echo "  -g GENERATOR"

}

CTYPE=gnu
BUILD_TYPE=Release
GENERATOR="Unix Makefiles"

while getopts ":g:" OPTION
do
  case $OPTION in
    g) GENERATOR=$OPTARG ;;
    *) usage; exit 1;
    ;;
  esac
done

shift $(($OPTIND - 1))

if [ $# -lt 1 ]  ; then
    usage
    exit
else
    CTYPE=$1
fi

case $CTYPE in
    "gnu")
        CC=`which mpicc`
        CXX=`which mpicxx`
        FC=`which mpif90`
        ##FCFLAGS="-g -O3 -cpp -Wconversion -pedantic -fbacktrace -fbounds-check -fno-range-check"
        CFLAGS="-g -O3 -DNDEBUG -std=c99"
        CXXFLAGS="-g -O3 -DNDEBUG"
        ;;
    "intel")
        CC=`which mpicc`
        CXX=`which mpicxx`
        FC=`which mpif90`
        ##FCFLAGS="-m64 -g -O3 -fpp -check all -gen-interfaces -warn interfaces"
        CFLAGS="-g -O3 -DNDEBUG -m64 -std=c99 -vec-report"
        CXXFLAGS="-g -O3 -DNDEBUG -m64 -vec-report"
        ;;
    "cray")
        CXX=`which CC`
        CC=`which cc`
        FC=`which ftn`
        CFLAGS="-g -O3 -DNDEBUG -std=c99"
        CXXFLAGS="-g -O3 -DNDEBUG"
        ##FCFLAGS="-g -O3 -cpp -Wconversion -pedantic -fbacktrace -fbounds-check -fno-range-check -I$HDF5_DIR/include/44"
        ;;
    *) usage; exit 1;
        ;;
esac

CMAKE_PREFIX_PATH=$MPICH_DIR:$HDF5_DIR:$GSL_DIR \
cmake -G "$GENERATOR" \
-DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
-DCMAKE_C_COMPILER:FILEPATH=$CC \
-DCMAKE_C_FLAGS_RELEASE:STRING="$CFLAGS" \
-DCMAKE_C_FLAGS_DEBUG:STRING="-g" \
-DCMAKE_CXX_COMPILER:FILEPATH=$CXX \
-DCMAKE_CXX_FLAGS_RELEASE:STRING="$CXXFLAGS" \
-DCMAKE_CXX_FLAGS_DEBUG:STRING="-g" \
-DUSE_MKL:BOOL=OFF \
-DUSE_IPP:BOOL=OFF \
-DBUILD_PLATFORM:STRING=Cray \
-DCPU_MHZ:STRING=2400 \
..

