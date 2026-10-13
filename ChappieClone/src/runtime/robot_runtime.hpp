#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace chappie {
struct Vec2 { double x{0}, y{0}; };
struct Pose2 { Vec2 position{}; double heading{0}; };
struct Twist2 { double linear{0}, angular{0}; };
struct RangeReading { double meters{0}; bool valid{false}; };
struct SensorSnapshot {
  std::uint64_t sequence{0};
  double battery_fraction{1.0};
  std::array<RangeReading, 8> ranges{};
  Pose2 pose{};
};
enum class AutonomyLevel : std::uint8_t { Manual=0, Assisted=1, Supervised=2, Autonomous=3 };
enum class ActionKind : std::uint8_t { Stop, Drive, Turn, Speak, MoveJoint };
struct Action {
  ActionKind kind{ActionKind::Stop};
  double a{0}, b{0};
  std::string text;
};
struct SafetyDecision { bool allowed{false}; std::string reason; };

class SecurityManager {
 public:
  explicit SecurityManager(AutonomyLevel level=AutonomyLevel::Assisted);
  void set_level(AutonomyLevel level);
  AutonomyLevel level() const;
  void deny_zone(std::string name);
  void allow_zone(const std::string& name);
  SafetyDecision authorize(const Action& action, bool confirmation, const std::string& zone) const;
 private:
  AutonomyLevel level_;
  std::vector<std::string> denied_zones_;
};

class OccupancyGrid {
 public:
  OccupancyGrid(int width, int height, double cell_size=1.0);
  int width() const; int height() const; double cell_size() const;
  bool in_bounds(int x,int y) const;
  bool blocked(int x,int y) const;
  void set_blocked(int x,int y,bool blocked=true);
  std::vector<Vec2> plan(Vec2 start, Vec2 goal) const;
 private:
  int width_,height_; double cell_size_; std::vector<std::uint8_t> cells_;
};

class RobotBody {
 public:
  explicit RobotBody(double battery_capacity=100.0);
  void command(Twist2 command);
  void emergency_stop();
  void tick(double dt_seconds);
  void set_obstacle_distance(double meters);
  Pose2 pose() const; Twist2 velocity() const;
  double battery_fraction() const;
  bool stopped() const;
 private:
  Pose2 pose_{}; Twist2 command_{}; double capacity_,charge_; double obstacle_m_{1000};
};

class MemoryStore {
 public:
  explicit MemoryStore(std::size_t capacity=512);
  void remember(std::string key,std::string value);
  std::optional<std::string> recall(const std::string& key) const;
  std::vector<std::pair<std::string,std::string>> recent(std::size_t count=20) const;
 private:
  std::size_t capacity_; std::deque<std::pair<std::string,std::string>> entries_;
};

class RobotRuntime {
 public:
  explicit RobotRuntime(AutonomyLevel level=AutonomyLevel::Assisted);
  SensorSnapshot sense() const;
  SafetyDecision act(const Action& action, bool confirmed=false, const std::string& zone="default");
  void tick(double dt_seconds);
  void set_obstacle_distance(double meters);
  RobotBody& body(); const RobotBody& body() const;
  SecurityManager& security(); MemoryStore& memory();
  const std::vector<std::string>& audit_log() const;
 private:
  RobotBody body_; SecurityManager security_; MemoryStore memory_;
  std::uint64_t sequence_{0}; std::vector<std::string> audit_;
};
}
