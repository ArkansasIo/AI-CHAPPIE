#pragma once
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace chappie {
struct Pose2 { double x{0},y{0},heading{0}; };
struct Action { enum class Kind { Stop,Drive,Turn,Speak }; Kind kind{Kind::Stop}; double value{0}; std::string text; };
enum class AutonomyLevel { Manual,Assisted,Supervised,Autonomous };
struct Decision { bool allowed; std::string reason; };
class SecurityManager {
 public: explicit SecurityManager(AutonomyLevel level=AutonomyLevel::Assisted):level_(level){}
 void set_level(AutonomyLevel l){level_=l;} void deny_zone(std::string z); void allow_zone(const std::string& z);
 Decision authorize(const Action&,bool confirmed,const std::string& zone) const;
 private: AutonomyLevel level_; std::vector<std::string> denied_;
};
class OccupancyGrid {
 public: OccupancyGrid(int width,int height,double cell=1.0); void set_blocked(int x,int y,bool blocked=true); bool blocked(int x,int y)const; std::vector<Pose2> plan(Pose2 start,Pose2 goal)const;
 private:int w_,h_;double cell_;std::vector<unsigned char> cells_;bool inside(int x,int y)const;
};
class RobotBody {
 public: void command(double linear,double angular=0);void stop();void tick(double dt);void set_obstacle_distance(double meters);Pose2 pose()const{return pose_;}double battery()const{return battery_;}bool stopped()const{return linear_==0&&angular_==0;}
 private:Pose2 pose_{};double linear_{0},angular_{0},battery_{1},obstacle_{1000};
};
class MemoryStore {
 public:explicit MemoryStore(std::size_t capacity=512);void remember(std::string key,std::string value);std::optional<std::string> recall(const std::string& key)const;
 private:std::size_t capacity_;std::deque<std::pair<std::string,std::string>> data_;
};
class RobotRuntime {
 public:explicit RobotRuntime(AutonomyLevel l=AutonomyLevel::Assisted):security_(l){} Decision act(const Action&,bool confirmed=false,const std::string& zone="default");void tick(double dt);Pose2 pose()const{return body_.pose();}double battery()const{return body_.battery();}void set_obstacle_distance(double m){body_.set_obstacle_distance(m);}SecurityManager& security(){return security_;}MemoryStore& memory(){return memory_;}const std::vector<std::string>& audit()const{return audit_;}
 private:RobotBody body_;SecurityManager security_;MemoryStore memory_;std::vector<std::string> audit_;
};
}
