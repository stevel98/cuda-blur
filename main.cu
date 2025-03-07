#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <string>

constexpr int BLUR_SIZE = 8;

// PMPP Edition 4, page 60
// TODO: Time and optimize for warps
__global__ void BlurKernel(uchar* in, uchar* out, int width, int height)
{   
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const auto in_bounds = [&] (int row, int col) {
        return col >= 0 && row >= 0 && col < width && row < height;
    };

    if (in_bounds(row, col)) {
        int pixel = 0;
        int num_pixels = 0;
        for (int col_idx = col - BLUR_SIZE; col_idx <= col + BLUR_SIZE; ++col_idx) {
            for (int row_idx = row - BLUR_SIZE; row_idx <= row + BLUR_SIZE; ++row_idx) {
                if (in_bounds(row_idx, col_idx)) {
                    pixel += in[row_idx * width + col_idx];
                    ++num_pixels;
                }
            }
        }
        out[row * width + col] = (pixel / num_pixels);
    }
}

std::string OutputFile(const std::string& s) {
    int idx = s.find(".");
    std::string prefix = s.substr(0, idx);
    std::string postfix = s.substr(idx, s.length());
    return prefix + "_Blurred" + postfix;
}

class ScopedTimer {
 public:
  ScopedTimer() {
    cudaEventCreate(&begin_);
    cudaEventCreate(&end_);
    cudaEventRecord(begin_);
  }

  ~ScopedTimer() {
    cudaEventRecord(end_);
    cudaEventSynchronize(begin_);
    cudaEventSynchronize(end_);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, begin_, end_);
    printf("Elapsed time: (%7.6f) s\n", elapsed_time_ms / 1000);
    // TODO: Calculate flops
  }

 private:
  cudaEvent_t begin_;
  cudaEvent_t end_;
};

int main() {    
    const std::string absolute_path = "/home/coder/volume/blur/Mona_Lisa.jpg"; 
    std::cout << "Path: " << absolute_path << "\n";
    std::cout << "BLUR_SIZE: " << BLUR_SIZE << "\n";

    cv::Mat input_mat = cv::imread(absolute_path, cv::IMREAD_GRAYSCALE);
    cv::Mat output_mat = cv::Mat(input_mat.size(), input_mat.type());
    std::cout << "input_mat.size() " << input_mat.size() << "\n";
    std::cout << "input_mat.height: " << input_mat.size().height << "\n";
    std::cout << "input_mat.width: " << input_mat.size().width << "\n";

    const size_t array_size = input_mat.total() * sizeof(uchar);
    uchar* d_input;
    uchar* d_output;
    cudaMalloc(&d_input, array_size);
    cudaMalloc(&d_output, array_size);

    cudaMemcpy(d_input, input_mat.ptr<uchar>(0), array_size, cudaMemcpyHostToDevice);
    // Is it necessary to copy the output mat over?
    cudaMemcpy(d_output, output_mat.ptr<uchar>(0), array_size, cudaMemcpyHostToDevice);

    const cv::Size size = input_mat.size();
    dim3 gridDim(ceil(size.width / 32.0), ceil(size.height / 32.0), 1);
    dim3 blockDim(32, 32, 1);  
    std::cout << "gridDim: " << gridDim.x << "," << gridDim.y << "," << gridDim.z << "\n";
    std::cout << "blockDim: " << blockDim.x << "," << blockDim.y << "," << blockDim.z << "\n";

    {
        ScopedTimer timer;
        BlurKernel <<<gridDim, blockDim>>> (d_input, d_output, size.width, size.height);
    }

    uchar* h_output = (uchar*) malloc(array_size);
    cudaMemcpy(h_output, d_output, array_size, cudaMemcpyDeviceToHost);
    cv::Mat output_mat_to_write(input_mat.size(), input_mat.type(), (void*) h_output);
    
    std::string output_file = OutputFile(absolute_path);
    std::cout << "Output file: " << output_file << "\n";
    if (!cv::imwrite(output_file, output_mat_to_write)) {
        std::cout << "failed to write image\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}