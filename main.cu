#include "opencv2/opencv.hpp"
#include "blur.h"

#include <string>

__global__ void blurKernel(uchar3* d_inputMat, uchar* d_kernelMat)
{
    return;
}

void blurCaller(const cv::Mat& inputMat, cv::Mat& kernelMat)
{
    // allocate device pointers
    uchar3 *d_inputMat;
    uchar  *d_kernelMat;
    cudaMalloc(&d_inputMat,  inputMat.total() * sizeof(uchar3));
    cudaMalloc(&d_kernelMat, kernelMat.total() * sizeof(uchar));

    // copy from host to device
    cudaMemcpy(d_inputMat, inputMat.ptr<uchar3>(0), inputMat.total() * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernelMat, kernelMat.ptr<uchar>(0), kernelMat.total() * sizeof(uchar), cudaMemcpyHostToDevice);

    // call CUDA kernel
    blurKernel <<<1, 1>>> (d_inputMat, d_kernelMat);

    // free
    cudaFree(d_inputMat);
    cudaFree(d_kernelMat);
}

int main() {
    constexpr std::string_view file = "file.txt";
    // input data
    cv::Mat inputMat(cv::Size(128, 128), CV_8UC3, cv::Scalar(100));
    cv::Mat kernelMat(cv::Size(16, 16), CV_8UC1, cv::Scalar(1));

    // call CUDA
    blurCaller(inputMat, kernelMat);

    std::cout << "done\n";
    return 0;
}