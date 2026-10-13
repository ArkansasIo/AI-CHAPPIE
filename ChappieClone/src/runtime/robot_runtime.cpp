#include "robot_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace chappie {
SecurityManager::SecurityManager(AutonomyLevel level):level_(level){}
void SecurityManager::set_level(AutonomyLevel level){level_=level;}
AutonomyLevel SecurityManager::level() const{return level_;}
void SecurityManager::deny_zone(std::string name){if(std::find(denied_zones_.begin(),denied_zones_.end(),name)==denied_zones_.end())denied_zones_.push_back(std::move(name));}
void SecurityManager::allow_zone(const std::string& name){denied_zones_.erase(std::remove(denied_zones_.begin(),denied_zones_.end(),name),denied_zones_.end());}
SafetyDecision SecurityManager::authorize(const Action& a,bool confirmation,const std::string& zone) const {
 if(std::find(denied_zones_.begin(),denied_zones_.end(),zone)!=denied_zones_.end())return {false,"target zone is denied"};
 if(a.kind==ActionKind::Stop)return {true,"emergency stop is always permitted"};
 if(level_==AutonomyLevel::Manual)return {false,"manual autonomy does not permit autonomous actions"};
 if((a.kind==ActionKind::MoveJoint||a.kind==ActionKind::Drive)&&level_!=AutonomyLevel::Autonomous&&!confirmation)return {false,"motion requires explicit confirmation at this autonomy level"};
 if((a.kind==ActionKind::MoveJoint||a.kind==ActionKind::Drive)&&level_==AutonomyLevel::Autonomous&&!confirmation)return {false,"motion must be confirmed by the runtime safety gate"};
 if(!std::isfinite(a.a)||!std::isfinite(a.b))return {false,"non-finite action parameter"};
 if(a.kind==ActionKind::Drive&&std::abs(a.a)>1.0)return {false,"linear command exceeds normalized range [-1,1]"};
 if(a.kind==ActionKind::Turn&&std::abs(a.a)>1.0)return {false,"turn command exceeds normalized range [-1,1]"};
 return {true,"authorized"};
}
OccupancyGrid::OccupancyGrid(int w,int h,double s):width_(w),height_(h),cell_size_(s),cells_(static_cast<std::size_t>(w)*static_cast<std::size_t>(h),0){if(w<=0||h<=0||!std::isfinite(s)||s<=0)throw std::invalid_argument("invalid occupancy grid dimensions or cell size");}
int OccupancyGrid::width()const{return width_;} int OccupancyGrid::height()const{return height_;} double OccupancyGrid::cell_size()const{return cell_size_;}
bool OccupancyGrid::in_bounds(int x,int y)const{return x>=0&&y>=0&&x<width_&&y<height_;}
bool OccupancyGrid::blocked(int x,int y)const{return !in_bounds(x,y)||cells_[static_cast<std::size_t>(y*width_+x)]!=0;}
void OccupancyGrid::set_blocked(int x,int y,bool b){if(!in_bounds(x,y))throw std::out_of_range("grid coordinate out of bounds");cells_[static_cast<std::size_t>(y*width_+x)]=b?1:0;}
std::vector<Vec2> OccupancyGrid::plan(Vec2 start,Vec2 goal)const{
 auto cell=[this](Vec2 p){return std::pair<int,int>{static_cast<int>(std::floor(p.x/cell_size_)),static_cast<int>(std::floor(p.y/cell_size_))};};
 auto [sx,sy]=cell(start);auto [gx,gy]=cell(goal);if(blocked(sx,sy)||blocked(gx,gy))return {};
 const int n=width_*height_, source=sy*width_+sx,target=gy*width_+gx;
 std::vector<double> g(static_cast<std::size_t>(n),std::numeric_limits<double>::infinity());std::vector<int> parent(static_cast<std::size_t>(n),-1);
 using Node=std::pair<double,int>;std::priority_queue<Node,std::vector<Node>,std::greater<Node>> q;g[source]=0;q.push({0,source});
 const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
 while(!q.empty()){auto [cost,u]=q.top();q.pop();if(cost!=g[u])continue;if(u==target)break;int x=u%width_,y=u/width_;
  for(int k=0;k<4;++k){int nx=x+dx[k],ny=y+dy[k];if(blocked(nx,ny))continue;int v=ny*width_+nx;double ng=cost+1;if(ng<g[v]){g[v]=ng;parent[v]=u;q.push({ng+std::abs(nx-gx)+std::abs(ny-gy),v});}}
 }
 if(source!=target&&parent[target]<0)return {};std::vector<Vec2> path;for(int u=target;u!=-1;u=parent[u]){path.push_back({(u%width_+0.5)*cell_size_,(u/width_+0.5)*cell_size_});if(u==source)break;}std::reverse(path.begin(),path.end());return path;
}
RobotBody::RobotBody(double cap):capacity_(cap),charge_(cap){if(!std::isfinite(cap)||cap<=0)throw std::invalid_argument("battery capacity must be positive");}
void RobotBody::command(Twist2 c){if(!std::isfinite(c.linear)||!std::isfinite(c.angular))throw std::invalid_argument("non-finite velocity");command_={std::clamp(c.linear,-1.0,1.0),std::clamp(c.angular,-1.0,1.0)};}
void RobotBody::emergency_stop(){command_={0,0};}
void RobotBody::tick(double dt){if(!std::isfinite(dt)||dt<0)throw std::invalid_argument("tick duration must be finite and nonnegative");if(charge_<=0){emergency_stop();return;}if(obstacle_m_<0.25&&command_.linear>0)command_.linear=0;pose_.heading=std::remainder(pose_.heading+command_.angular*dt,6.283185307179586);pose_.position.x+=std::cos(pose_.heading)*command_.linear*dt;pose_.position.y+=std::sin(pose_.heading)*command_.linear*dt;charge_=std::max(0.0,charge_-dt*(0.02+0.1*std::abs(command_.linear)+0.03*std::abs(command_.angular)));if(charge_<=0)emergency_stop();}
void RobotBody::set_obstacle_distance(double m){if(!std::isfinite(m)||m<0)throw std::invalid_argument("obstacle distance must be nonnegative");obstacle_m_=m;}
Pose2 RobotBody::pose()const{return pose_;} Twist2 RobotBody::velocity()const{return command_;} double RobotBody::battery_fraction()const{return charge_/capacity_;}bool RobotBody::stopped()const{return command_.linear==0&&command_.angular==0;}
MemoryStore::MemoryStore(std::size_t c):capacity_(c){if(c==0)throw std::invalid_argument("memory capacity must be positive");}
void MemoryStore::remember(std::string k,std::string v){auto it=std::find_if(entries_.begin(),entries_.end(),[&](const auto& e){return e.first==k;});if(it!=entries_.end())entries_.erase(it);entries_.emplace_back(std::move(k),std::move(v));while(entries_.size()>capacity_)entries_.pop_front();}
std::optional<std::string> MemoryStore::recall(const std::string& k)const{for(auto i=entries_.rbegin();i!=entries_.rend();++i)if(i->first==k)return i->second;return std::nullopt;}
std::vector<std::pair<std::string,std::string>> MemoryStore::recent(std::size_t n)const{std::vector<std::pair<std::string,std::string>> out;auto begin=entries_.size()>n?entries_.size()-n:0;for(std::size_t i=begin;i<entries_.size();++i)out.push_back(entries_[i]);return out;}
RobotRuntime::RobotRuntime(AutonomyLevel l):security_(l){}
SensorSnapshot RobotRuntime::sense()const{SensorSnapshot s;s.sequence=sequence_;s.battery_fraction=body_.battery_fraction();s.pose=body_.pose();return s;}
SafetyDecision RobotRuntime::act(const Action& a,bool confirmed,const std::string& zone){auto d=security_.authorize(a,confirmed,zone);std::ostringstream line;line<<"action="<<static_cast<int>(a.kind)<<" allowed="<<(d.allowed?"true":"false")<<" reason="<<d.reason;audit_.push_back(line.str());if(audit_.size()>2048)audit_.erase(audit_.begin());if(!d.allowed)return d;switch(a.kind){case ActionKind::Stop:body_.emergency_stop();break;case ActionKind::Drive:body_.command({a.a,0});break;case ActionKind::Turn:body_.command({0,a.a});break;case ActionKind::Speak:memory_.remember("last_speech",a.text);break;case ActionKind::MoveJoint:return {false,"virtual differential-drive body does not expose joints"};}return d;}
void RobotRuntime::tick(double dt){body_.tick(dt);++sequence_;}void RobotRuntime::set_obstacle_distance(double m){body_.set_obstacle_distance(m);}RobotBody& RobotRuntime::body(){return body_;}const RobotBody& RobotRuntime::body()const{return body_;}SecurityManager& RobotRuntime::security(){return security_;}MemoryStore& RobotRuntime::memory(){return memory_;}const std::vector<std::string>& RobotRuntime::audit_log()const{return audit_;}
}
