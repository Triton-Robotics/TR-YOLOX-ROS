#pragma once 
#include "yolox_ros_cpp/yolox_ros_cpp.hpp"
#include <condition_variable>
#include <mutex>

class SingleFrameBuffer {

private:
    cv::Mat frame;
    std::mutex mtx;
    std::condition_variable not_empty;

public:
    void push(const cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mtx);
        this->frame = frame;
        not_empty.notify_one();
    }
    cv::Mat pop() {
        cv::Mat output_frame;
        {
            std::unique_lock<std::mutex> lock(mtx);
            not_empty.wait(lock, [this] {return !frame.empty()});

            output_frame = std::move(this->frame);
            this->frame.release();
        }
        return output_frame;
    }
};