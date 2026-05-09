#pragma once

#include <chrono>
#include <cmath>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "std_msgs/msg/float32.hpp"

#include "yolox_cpp/utils.hpp"
#include "yolox_cpp/yolox.hpp"
#include "yolox_param/yolox_param.hpp"

#include "shm/SharedDetWithImage.h"
#include "shm/SharedImage.h"

#include "tr_messages/msg/det_with_img.hpp"

#include "tr_debug/debug.hpp"

#include "nvtx_pipeline.hpp"

namespace yolox_ros_cpp {
class YoloXNode : public rclcpp::Node {
  public:
    YoloXNode(const rclcpp::NodeOptions &);

  private:
    void onInit();
    // void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &); Deprecated
    void sharedMemoryImageCallback(); // New callback for shared memory polling
    static vision_msgs::msg::Detection2DArray
    objects_to_detection2d(const std::vector<yolox_cpp::Object> &, const std_msgs::msg::Header &);
    static Detection2DArray objects_to_shm_detection2darray(const std::vector<yolox_cpp::Object> &,
                                                            const long &);

  protected:
    std::shared_ptr<yolox_parameters::ParamListener> param_listener_;
    yolox_parameters::Params params_;

  private:
    bool init;

    bool publishToRos = false;

    cudaStream_t stream_;

    size_t input_bytes;
    size_t output_bytes;

    uchar3 *d_image_;
    float *d_output_;

    std::unique_ptr<yolox_cpp::AbcYoloX> yolox_;
    std::vector<std::string> class_names_;

    rclcpp::CallbackGroup::SharedPtr callback_group_reentrant_;
    std::shared_ptr<rclcpp::SubscriptionOptions> sub_options_;

    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::TimerBase::SharedPtr shm_timer_; // Timer for polling shared memory
    image_transport::Subscriber sub_image_;

    // Shared memory components
    std::unique_ptr<SharedImageReader> sharedImageReader_;
    int last_frame_; // Track last processed frame
    std::unique_ptr<SharedDetWithImageWriter> sharedDetWriter_;
    struct timespec curr_ts_;

    rclcpp::Publisher<tr_messages::msg::DetWithImg>::SharedPtr pub_detection2d_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_dets_image;

    // profiler latency publishers
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_zc_latency_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_latency_; // YOLOx callback latency
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr
        pub_cum_yolox_latency_; // camera grab -> end yolox callback

    image_transport::Publisher pub_image_;
    std::unique_ptr<PipelineTracer> tracer_;
};
} // namespace yolox_ros_cpp
