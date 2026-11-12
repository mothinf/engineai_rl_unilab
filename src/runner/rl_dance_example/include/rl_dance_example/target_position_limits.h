#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace runner::wbt {

// Dance-owned position target contract. This is not a parallel-ankle
// reachability test or a limit on the downstream motor torque.
class TargetPositionLimits {
 public:
  void Configure(int joints, const Eigen::VectorXi& mapping,
                 const std::optional<Eigen::VectorXd>& lower,
                 const std::optional<Eigen::VectorXd>& upper) {
    joints_ = 0;  // Failed reconfiguration must not leave old limits usable.
    if (joints <= 0 || joints > 64 || mapping.size() == 0)
      throw std::invalid_argument("invalid target joint dimensions");
    uint64_t seen = 0;
    for (int i = 0; i < mapping.size(); ++i) {
      const int j = mapping[i];
      if (j < 0 || j >= joints || (seen & (uint64_t{1} << j)))
        throw std::invalid_argument("invalid or duplicate target joint mapping");
      seen |= uint64_t{1} << j;
    }
    if (lower.has_value() != upper.has_value())
      throw std::invalid_argument("target lower and upper limits must both be provided");
    enabled_ = lower.has_value();
    if (enabled_) {
      if (lower->size() != mapping.size() || upper->size() != mapping.size() ||
          !lower->allFinite() || !upper->allFinite() ||
          (lower->array() >= upper->array()).any())
        throw std::invalid_argument("invalid target position limits");
      lower_ = *lower;
      upper_ = *upper;
    }
    mapping_ = mapping;
    joints_ = joints;
  }

  bool Apply(Eigen::VectorXd& target, uint64_t& clipped_mask) const {
    clipped_mask = 0;
    // Validate before clamp: Eigen min/max must not hide NaN/Inf.
    if (joints_ == 0 || target.size() != joints_ || !target.allFinite()) return false;
    if (!enabled_) return true;
    for (int i = 0; i < mapping_.size(); ++i) {
      const int j = mapping_[i];
      const double bounded = std::min(std::max(target[j], lower_[i]), upper_[i]);
      if (bounded != target[j]) clipped_mask |= uint64_t{1} << j;
      target[j] = bounded;
    }
    return true;
  }

 private:
  int joints_ = 0;
  bool enabled_ = false;
  Eigen::VectorXi mapping_;
  Eigen::VectorXd lower_, upper_;
};

}  // namespace runner::wbt
