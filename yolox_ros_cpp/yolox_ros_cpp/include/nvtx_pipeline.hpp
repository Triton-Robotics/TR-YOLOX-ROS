// nvtx_pipeline.hpp
#pragma once
#include <builtin_interfaces/msg/detail/time__struct.hpp>
#include <nvtx3/nvtx3.hpp>
#include <string>
#include <unordered_map>

class PipelineTracer {
public:
  explicit PipelineTracer(const std::string &node_name)
      : node_name_(node_name),
        domain_(nvtxDomainCreateA("ros2_image_pipeline")) {}

  void begin(uint64_t pipeline_id) {
    nvtxEventAttributes_t attrs = {};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.colorType = NVTX_COLOR_ARGB;
    attrs.color = colorForNode(node_name_);
    attrs.message.ascii = node_name_.c_str();
    // Attach the shared pipeline ID as payload — Nsight uses this to correlate
    attrs.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attrs.payload.ullValue = pipeline_id;

    auto range = nvtxDomainRangeStartEx(domain_, &attrs);
    ranges_[pipeline_id] = range;
  }

  void end(uint64_t pipeline_id) {
    auto it = ranges_.find(pipeline_id);
    if (it != ranges_.end()) {
      nvtxDomainRangeEnd(domain_, it->second);
      ranges_.erase(it);
    }
  }

  uint64_t stampToId(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<uint64_t>(stamp.sec) * 1'000'000'000ULL +
           static_cast<uint64_t>(stamp.nanosec);
  }

private:
  uint32_t colorForNode(const std::string &name) {
    // Stable color per node name
    size_t h = std::hash<std::string>{}(name);
    return 0xFF000000 | (h & 0x00FFFFFF);
  }

  std::string node_name_;
  nvtxDomainHandle_t domain_;
  std::unordered_map<uint64_t, nvtxRangeId_t> ranges_;
};