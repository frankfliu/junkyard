#!/usr/bin/env bash
set -exo pipefail

#############################################
# build opencv from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/opencv" ]]; then
        cd ${WORK_DIR}/src/opencv
        set +e
        rm -rf build
    fi
    exit
fi

OPENCV_INSTALL_DIR=${WORK_DIR}/build/opencv-${OPENCV_VERSION}
OPENBLAS_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name OpenBLAS-*)
ZLIB_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name zlib-*)
TIFF_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name libtiff-*)
PNG_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name libpng-*)
JPEG_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name libjpeg-turbo-*)
EIGEN_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name eigen-*)

mkdir -p ${OPENCV_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "opencv" ]; then
    git clone https://github.com/opencv/opencv.git
fi
cd opencv

git fetch
git checkout ${OPENCV_VERSION}

echo "Building OpenCV ${OPENCV_VERSION} ..."

mkdir -p build
cd build

# Copying patch header file to opencv
cp -f ${WORK_DIR}/patch/opencv_lapack.h .

if [[ "${PLATFORM}" == "linux" ]]; then
    OPENCV_LAPACK_OPTIONS=" \
          -D OpenBLAS_HOME=${WORK_DIR}/src/OpenBLAS \
          -D OpenBLAS_INCLUDE_DIR=${OPENBLAS_INSTALL_DIR}/include \
          -D OpenBLAS_LIB=${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a \
          -D LAPACK_INCLUDE_DIR=${OPENBLAS_INSTALL_DIR}/include \
          -D LAPACK_LINK_LIBRARIES=${OPENBLAS_INSTALL_DIR}/lib/ \
          -D LAPACK_LIBRARIES=${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a \
          -D LAPACK_CBLAS_H='cblas.h' \
          -D LAPACK_LAPACKE_H='lapacke.h' \
          -D LAPACK_IMPL='OpenBLAS' \
          -D HAVE_LAPACK=1"
fi

export CPATH=${OPENBLAS_INSTALL_DIR}/include:$CPATH

cmake -q \
    -D WITH_PTHREADS_PF=ON \
    -D WITH_LAPACK=ON \
    -D WITH_EIGEN=ON \
    -D WITH_PNG=ON \
    -D WITH_JPEG=ON \
    -D WITH_TIFF=ON \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D WITH_1394=OFF \
    -D WITH_ARAVIS=OFF \
    -D WITH_AVFOUNDATION=OFF \
    -D WITH_CAROTENE=OFF \
    -D WITH_CLP=OFF \
    -D WITH_CSTRIPES=OFF \
    -D WITH_CPUFEATURES=OFF \
    -D WITH_CUBLAS=OFF \
    -D WITH_CUDA=OFF \
    -D WITH_CUFFT=OFF \
    -D WITH_DIRECTX=OFF \
    -D WITH_DSHOW=OFF \
    -D WITH_FFMPEG=OFF \
    -D WITH_GDAL=OFF \
    -D WITH_GDCM=OFF \
    -D WITH_GIGEAPI=OFF \
    -D WITH_GPHOTO2=OFF \
    -D WITH_GSTREAMER=OFF \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_GTK=OFF \
    -D WITH_GTK_2_X=OFF \
    -D WITH_HALIDE=OFF \
    -D WITH_IMAGEIO=OFF \
    -D WITH_IMGCODEC_HDR=OFF \
    -D WITH_IMGCODEC_PXM=OFF \
    -D WITH_IMGCODEC_SUNRASTER=OFF \
    -D WITH_INF_ENGINE=OFF \
    -D WITH_INTELPERC=OFF \
    -D WITH_IPP=OFF \
    -D WITH_IPP_A=OFF \
    -D WITH_ITT=OFF \
    -D WITH_JASPER=OFF \
    -D WITH_LIBREALSENSE=OFF \
    -D WITH_LIBV4L=OFF \
    -D WITH_MATLAB=OFF \
    -D WITH_MFX=OFF \
    -D WITH_MSMF=OFF \
    -D WITH_NVCUVID=OFF \
    -D WITH_OPENCL=OFF \
    -D WITH_OPENCLAMDBLAS=OFF \
    -D WITH_OPENCLAMDFFT=OFF \
    -D WITH_OPENCL_SVM=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_OPENGL=OFF \
    -D WITH_OPENMP=OFF \
    -D WITH_OPENNI=OFF \
    -D WITH_OPENNI2=OFF \
    -D WITH_OPENVX=OFF \
    -D WITH_PROTOBUF=OFF \
    -D WITH_PVAPI=OFF \
    -D WITH_QT=OFF \
    -D WITH_QTKIT=OFF \
    -D WITH_QUICKTIME=OFF \
    -D WITH_TBB=OFF \
    -D WITH_UNICAP=OFF \
    -D WITH_V4L=OFF \
    -D WITH_VA=OFF \
    -D WITH_VA_INTEL=OFF \
    -D WITH_VFW=OFF \
    -D WITH_VTK=OFF \
    -D WITH_WEBP=OFF \
    -D WITH_WIN32UI=OFF \
    -D WITH_XIMEA=OFF \
    -D WITH_XINE=OFF \
    -D BUILD_ANDROID_EXAMPLES=OFF \
    -D BUILD_ANDROID_PROJECTS=OFF \
    -D BUILD_ANDROID_SERVICE=OFF \
    -D BUILD_CUDA_STUBS=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_FAT_JAVA_LIB=OFF \
    -D BUILD_IPP_IW=OFF \
    -D BUILD_ITT_IW=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_JASPER=OFF \
    -D BUILD_JPEG=OFF \
    -D BUILD_OPENEXR=OFF \
    -D BUILD_PACKAGE=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_PNG=OFF \
    -D BUILD_SHARED_LIBS=OFF \
    -D BUILD_TBB=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_TIFF=OFF \
    -D BUILD_WEBP=OFF \
    -D BUILD_WITH_DEBUG_INFO=OFF \
    -D BUILD_WITH_DYNAMIC_IPP=OFF \
    -D BUILD_WITH_STATIC_CRT=OFF \
    -D BUILD_ZLIB=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_opencv_aruco=OFF \
    -D BUILD_opencv_calib3d=OFF \
    -D BUILD_opencv_contrib=OFF \
    -D BUILD_opencv_dnn=OFF \
    -D BUILD_opencv_features2d=OFF \
    -D BUILD_opencv_flann=OFF \
    -D BUILD_opencv_gpu=OFF \
    -D BUILD_opencv_gpuarithm=OFF \
    -D BUILD_opencv_gpubgsegm=OFF \
    -D BUILD_opencv_gpucodec=OFF \
    -D BUILD_opencv_gpufeatures2d=OFF \
    -D BUILD_opencv_gpufilters=OFF \
    -D BUILD_opencv_gpuimgproc=OFF \
    -D BUILD_opencv_gpulegacy=OFF \
    -D BUILD_opencv_gpuoptflow=OFF \
    -D BUILD_opencv_gpustereo=OFF \
    -D BUILD_opencv_gpuwarping=OFF \
    -D BUILD_opencv_highgui=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_js=OFF \
    -D BUILD_opencv_ml=OFF \
    -D BUILD_opencv_ml=OFF \
    -D BUILD_opencv_nonfree=OFF \
    -D BUILD_opencv_objdetect=OFF \
    -D BUILD_opencv_photo=OFF \
    -D BUILD_opencv_python=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_superres=OFF \
    -D BUILD_opencv_video=OFF \
    -D BUILD_opencv_videoio=OFF \
    -D BUILD_opencv_videostab=OFF \
    -D BUILD_opencv_viz=OFF \
    -D BUILD_opencv_world=OFF \
    -D OPENCV_LIB_INSTALL_PATH=lib \
    -D OPENCV_INCLUDE_INSTALL_PATH=include \
    -D CMAKE_LIBRARY_PATH=${OPENBLAS_INSTALL_DIR}/lib \
    -D CMAKE_INCLUDE_PATH=${OPENBLAS_INSTALL_DIR}/include \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR} \
    -D ZLIB_LIBRARY=${ZLIB_INSTALL_DIR}/lib/libz.a \
    -D ZLIB_INCLUDE_DIR=${ZLIB_INSTALL_DIR}/include \
    -D PNG_LIBRARY=${PNG_INSTALL_DIR}/lib/libpng.a \
    -D PNG_PNG_INCLUDE_DIR=${PNG_INSTALL_DIR}/include \
    -D TIFF_LIBRARY=${TIFF_INSTALL_DIR}/lib/libtiff.a \
    -D TIFF_INCLUDE_DIR=${TIFF_INSTALL_DIR}/include \
    -D JPEG_LIBRARY=${JPEG_INSTALL_DIR}/lib/libjpeg.a \
    -D JPEG_INCLUDE_DIR=${JPEG_INSTALL_DIR}/include \
    -D EIGEN_INCLUDE_PATH=${EIGEN_INSTALL_DIR}/include/eigen3 \
    ${OPENCV_LAPACK_OPTIONS} \
    ..

make -j ${NUM_PROC}
make install

# this is a hack to use the compatibility header
cat ${OPENCV_INSTALL_DIR}/include/opencv2/imgcodecs/imgcodecs_c.h >> ${OPENCV_INSTALL_DIR}/include/opencv2/imgcodecs.hpp

tar cvfz ${OPENCV_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${OPENCV_INSTALL_DIR} .

cd ${WORK_DIR}