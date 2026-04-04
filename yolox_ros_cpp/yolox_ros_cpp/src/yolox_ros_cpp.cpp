#include "yolox_ros_cpp/yolox_ros_cpp.hpp"

namespace yolox_ros_cpp {
YoloXNode::YoloXNode(const rclcpp::NodeOptions &options) : Node("yolox_ros_cpp", options) {
    using namespace std::chrono_literals; // NOLINT
    this->init_timer_ = this->create_wall_timer(0s, std::bind(&YoloXNode::onInit, this));
    this->declare_parameter("publish_to_ros", false);
}

void YoloXNode::onInit() {
    this->init = true;
    this->publishToRos = this->get_parameter("publish_to_ros").as_bool();
    this->d_image_ = nullptr;
    this->d_output_ = nullptr;
    cudaStreamCreate(&this->stream_);
    rclcpp::on_shutdown([this]() {
        cudaStreamDestroy(this->stream_);
        cudaFree(this->d_image_);
        cudaFree(this->d_output_);
        this->stream_ = nullptr;
    });
    this->init_timer_->cancel();
    this->param_listener_ =
        std::make_shared<yolox_parameters::ParamListener>(this->get_node_parameters_interface());
    this->params_ = this->param_listener_->get_params();
    this->callback_group_reentrant_ =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    this->sub_options_ = std::make_shared<rclcpp::SubscriptionOptions>();
    this->sub_options_->callback_group = callback_group_reentrant_;
    this->sub_options_->use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    if (this->params_.imshow_isshow) {
        cv::namedWindow("yolox", cv::WINDOW_AUTOSIZE);
    }

    if (this->params_.class_labels_path != "") {
        RCLCPP_INFO(this->get_logger(), "read class labels from '%s'",
                    this->params_.class_labels_path.c_str());
        this->class_names_ =
            yolox_cpp::utils::read_class_labels_file(this->params_.class_labels_path);
    } else {
        this->class_names_ = yolox_cpp::COCO_CLASSES;
    }

    if (this->params_.model_type == "tensorrt") {
#ifdef ENABLE_TENSORRT
        RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXTensorRT>(
            this->params_.model_path, this->params_.tensorrt_device, this->params_.nms,
            this->params_.conf, this->params_.model_version, this->params_.num_classes,
            this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with TensorRT");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "openvino") {
#ifdef ENABLE_OPENVINO
        RCLCPP_INFO(this->get_logger(), "Model Type is OpenVINO");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXOpenVINO>(
            this->params_.model_path, this->params_.openvino_device, this->params_.nms,
            this->params_.conf, this->params_.model_version, this->params_.num_classes,
            this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with OpenVINO");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "onnxruntime") {
#ifdef ENABLE_ONNXRUNTIME
        RCLCPP_INFO(this->get_logger(), "Model Type is ONNXRuntime");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXONNXRuntime>(
            this->params_.model_path, this->params_.onnxruntime_intra_op_num_threads,
            this->params_.onnxruntime_inter_op_num_threads, this->params_.onnxruntime_use_cuda,
            this->params_.onnxruntime_device_id, this->params_.onnxruntime_use_parallel,
            this->params_.nms, this->params_.conf, this->params_.model_version,
            this->params_.num_classes, this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with ONNXRuntime");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "tflite") {
#ifdef ENABLE_TFLITE
        RCLCPP_INFO(this->get_logger(), "Model Type is tflite");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXTflite>(
            this->params_.model_path, this->params_.tflite_num_threads, this->params_.nms,
            this->params_.conf, this->params_.model_version, this->params_.num_classes,
            this->params_.p6, this->params_.is_nchw);
#else
        RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with tflite");
        rclcpp::shutdown();
#endif
    }
    RCLCPP_INFO(this->get_logger(), "model loaded");

    // Create publishers
    this->pub_detection2d_ = this->create_publisher<tr_messages::msg::DetWithImg>(
        this->params_.publish_detwithimg_topic_name, 10);

    this->pub_latency_ = this->create_publisher<std_msgs::msg::Float32>("yolox_latency_ms", 10);

    this->pub_zc_latency_ =
        this->create_publisher<std_msgs::msg::Float32>("cv_yolox_start_latency_ms", 10);

    this->pub_cum_yolox_latency_ =
        this->create_publisher<std_msgs::msg::Float32>("cv_yolox_end_latency_ms", 10);

    if (this->params_.publish_resized_image) {
        this->pub_image_ =
            image_transport::create_publisher(this, this->params_.publish_image_topic_name);
    }

    // Initialize SharedImageReader for camera_image shared memory
    this->sharedImageReader_ =
        std::make_unique<SharedImageReader>("camera_image", 1200, 1920, 3, "CV_8U");
    this->last_frame_ = -1; // Initialize to -1 to process first frame

    // Create timer for polling shared memory
    using namespace std::chrono_literals;
    this->shm_timer_ =
        this->create_wall_timer(0.1ms, std::bind(&YoloXNode::sharedMemoryImageCallback, this));

    RCLCPP_INFO(this->get_logger(), "Using shared memory input - SharedImageReader initialized "
                                    "for camera_image");

    this->sharedDetWriter_ = std::make_unique<SharedDetWithImageWriter>("yolox_det_with_image",
                                                                        1200, 1920, 3, "CV_8U");
}

void YoloXNode::sharedMemoryImageCallback() {
    // Check if SharedImageReader is initialized and ready
    if (!this->sharedImageReader_ || !this->sharedImageReader_->isInitialized()) {
        RCLCPP_INFO(this->get_logger(), "Shared memory reader not initialized.");
        return;
    }

    cv::Mat image;
    int current_frame = this->last_frame_;

    // Try to read new image from shared memory
    if (!this->sharedImageReader_->readImage(image, current_frame)) {
        return; // No new frame available
    }

    long timeGrabbed = this->sharedImageReader_->getTimeStamp();

    // Update last processed frame
    this->last_frame_ = current_frame;

    auto now_noninf = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();

    // Convert the current time point to nanoseconds since the epoch
    auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    auto image_zc_time = (now_ns - timeGrabbed) / 1000000.0F;

    // Initialization for CUDA memory (same as ROS callback)
    if (this->init) {
        this->input_bytes = sizeof(uchar3) * image.cols * image.rows;
        cudaMallocManaged(reinterpret_cast<void **>(&this->d_image_), this->input_bytes,
                          cudaMemAttachHost);
        this->output_bytes = sizeof(float) * 416 * 416 * 3;
        cudaMallocManaged(reinterpret_cast<void **>(&this->d_output_), this->output_bytes,
                          cudaMemAttachHost);
        this->init = false;
    }

    // Copy image data to CUDA memory
    auto copy_start = std::chrono::high_resolution_clock::now();
    std::memcpy(this->d_image_, image.data, this->input_bytes);
    auto copy_end = std::chrono::high_resolution_clock::now();

    // Run YOLOX inference
    auto objects = this->yolox_->inference(image, this->d_image_, this->d_output_, this->stream_);

    RCLCPP_INFO(this->get_logger(), "%zu objects detected", objects.size());

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - now);

    // Show image if enabled
    if (this->params_.imshow_isshow) {
        yolox_cpp::utils::draw_objects(image, objects, this->class_names_);
        cv::imshow("yolox", image);
        if (cv::waitKey(1) == 27) {
            rclcpp::shutdown();
        }
    }

    // Publish detections if any found
    if (!this->pub_detection2d_) {
        RCLCPP_ERROR(this->get_logger(), "pub_detection2d_ is nullptr");
        return;
    }

    // Create header with shared memory timestamp
    std_msgs::msg::Header header;
    header.stamp = rclcpp::Time(timeGrabbed);
    header.frame_id = "camera_optical_frame";

    vision_msgs::msg::Detection2DArray detections = objects_to_detection2d(objects, header);

    curr_ts_.tv_nsec = this->now().nanoseconds();
    Detection2DArray shared_detections = objects_to_shm_detection2darray(objects, curr_ts_.tv_nsec);

    if (!detections.detections.empty()) {
        if (this->publishToRos) {
            tr_messages::msg::DetWithImg detwithimg;
            detwithimg.image = *cv_bridge::CvImage(header, "bgr8", image).toImageMsg();  // Copy unavoidable due to const shared
            detwithimg.detection_info.detections = detections.detections;
            this->pub_detection2d_->publish(detwithimg);
        }
        // rewrite the image that was received from cam node
        this->sharedDetWriter_->writeDetWithImage(image, shared_detections, timeGrabbed);

        // RCLCPP_INFO(this->get_logger(), "Published %zu detections from shared
        // memory frame %d",
        //             detections.detections.size(), current_frame);
    } else {
        RCLCPP_DEBUG(this->get_logger(), "No detections from shared memory frame %d",
                     current_frame);
    }
    std::chrono::system_clock::time_point end_yolo = std::chrono::system_clock::now();
    // Convert the current time point to nanoseconds since the epoch
    auto end_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_yolo.time_since_epoch()).count();

    // time from beginning to end of yolox callback
    float yolox_time = (end_ns - now_ns) / 1000000.0F;
    // total time from image grabbed to end of yolox
    float cumulative_yolox_time = (end_ns - timeGrabbed) / 1000000.0F;
    // RCLCPP_INFO(this->get_logger(), "yolox time + ipc =  %0.3f + %0.3f =
    // %0.3f, raw %0.3f",
    //     yolox_time, image_zc_time, cumulative_yolox_time);

    // zero copy latency: camera node to yolox
    std_msgs::msg::Float32 zc_latency_msg;
    zc_latency_msg.data = image_zc_time;
    // yolox callback latency: yolox callback
    std_msgs::msg::Float32 yolox_latency_msg;
    yolox_latency_msg.data = yolox_time;
    // zero copy latency: camera node to yolox
    std_msgs::msg::Float32 cum_yolox_latency_msg;
    cum_yolox_latency_msg.data = cumulative_yolox_time;

    // Publish latency information
    auto end_noninf = std::chrono::system_clock::now();
    auto elapsed_noninf =
        std::chrono::duration_cast<std::chrono::microseconds>(end_noninf - now_noninf);

    std_msgs::msg::Float32 latency_msg;
    latency_msg.data = static_cast<float>(elapsed_noninf.count());
    this->pub_latency_->publish(latency_msg);
    this->pub_zc_latency_->publish(zc_latency_msg);
    this->pub_cum_yolox_latency_->publish(cum_yolox_latency_msg);
}

vision_msgs::msg::Detection2DArray
YoloXNode::objects_to_detection2d(const std::vector<yolox_cpp::Object> &objects,
                                  const std_msgs::msg::Header &header) {
    vision_msgs::msg::Detection2DArray detection2d;
    detection2d.header = header;
    for (const auto &obj : objects) {
        vision_msgs::msg::Detection2D det;
        det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
        det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;

        det.results.resize(1);
        det.results[0].hypothesis.class_id = std::to_string(obj.label);
        det.results[0].hypothesis.score = obj.prob;
        detection2d.detections.emplace_back(det);
    }
    return detection2d;
}

Detection2DArray
YoloXNode::objects_to_shm_detection2darray(const std::vector<yolox_cpp::Object> &objects,
                                           const long &ns) {
    Detection2DArray detections_array;
    detections_array.timestamp = ns;

    unsigned int num_detections = 0;
    for (const auto &obj : objects) {
        Detection2D det;
        det.bbox.center_x = obj.rect.x + obj.rect.width / 2;
        det.bbox.center_y = obj.rect.y + obj.rect.height / 2;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;

        det.score = obj.prob;
        detections_array.detections[num_detections] = det;
        num_detections++;
    }
    detections_array.num_detections = num_detections;
    return detections_array;
}

} // namespace yolox_ros_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolox_ros_cpp::YoloXNode)
