#include "robot_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
namespace chappie {
void SecurityManager::deny_zone(std::string z){if(std::find(denied_.begin(),denied_.end(),z)==denied_.end())denied_.push_back(std::move(z));}
void SecurityManager::allow_zone(const std::string& z){denied_.erase(std::remove(denied_.begin(),denied_.end(),z),denied_.end());}
Decision SecurityManager::authorize(const Action&a,bool confirmed,const std::string&zone)const{
 if(a.kind==Action::Kind::Stop)return {true,"stop always allowed"};
 if(std::find(denied_.begin(),denied_.end(),zone)!=denied_.end())return {false,"zone denied"};
 if(level_==AutonomyLevel::Manual)return {false,"manual mode blocks autonomous commands"};
 if(!std::isfinite(a.value))return {false,"non-finite command"};
 if((a.kind==Action::Kind::Drive||a.kind==Action::Kind::Turn)&&std::abs(a.value)>1)return {false,"command outside normalized [-1,1]"};
 if((a.kind==Action::Kind::Drive||a.kind==Action::Kind::Turn)&&!confirmed)return {false,"motion requires explicit confirmation"};
 return {true,"authorized"};
}
OccupancyGrid::OccupancyGrid(int w,int h,double c):w_(w),h_(h),cell_(c),cells_(static_cast<std::size_t>(w)*h,0){if(w<=0||h<=0||!std::isfinite(c)||c<=0)throw std::invalid_argument("invalid grid dimensions");}
bool OccupancyGrid::inside(int x,int y)const{return x>=0&&y>=0&&x<w_&&y<h_;}
bool OccupancyGrid::blocked(int x,int y)const{return !inside(x,y)||cells_[static_cast<std::size_t>(y*w_+x)]!=0;}
void OccupancyGrid::set_blocked(int x,int y,bool b){if(!inside(x,y))throw std::out_of_range("grid coordinate");cells_[static_cast<std::size_t>(y*w_+x)]=b?1:0;}
std::vector<Pose2> OccupancyGrid::plan(Pose2 s,Pose2 g)const{
 int sx=static_cast<int>(std::floor(s.x/cell_)),sy=static_cast<int>(std::floor(s.y/cell_)),gx=static_cast<int>(std::floor(g.x/cell_)),gy=static_cast<int>(std::floor(g.y/cell_));
 if(blocked(sx,sy)||blocked(gx,gy))return {};int n=w_*h_,src=sy*w_+sx,dst=gy*w_+gx;std::vector<int>d(n,-1),prev(n,-1);std::queue<int>q;q.push(src);d[src]=0;int dx[]={1,-1,0,0},dy[]={0,0,1,-1};
 while(!q.empty()){int u=q.front();q.pop();if(u==dst)break;int x=u%w_,y=u/w_;for(int k=0;k<4;++k){int xx=x+dx[k],yy=y+dy[k],v=yy*w_+xx;if(!blocked(xx,yy)&&d[v]<0){d[v]=d[u]+1;prev[v]=u;q.push(v);}}}
 if(d[dst]<0)return {};std::vector<Pose2>p;for(int u=dst;u>=0;u=prev[u]){p.push_back({(u%w_+.5)*cell_,(u/w_+.5)*cell_,0});if(u==src)break;}std::reverse(p.begin(),p.end());return p;
}
void RobotBody::command(double l,double a){if(!std::isfinite(l)||!std::isfinite(a))throw std::invalid_argument("non-finite velocity");linear_=std::clamp(l,-1.0,1.0);angular_=std::clamp(a,-1.0,1.0);}
void RobotBody::stop(){linear_=angular_=0;}
void RobotBody::tick(double dt){if(!std::isfinite(dt)||dt<0)throw std::invalid_argument("dt must be nonnegative");if(battery_<=0){stop();return;}if(obstacle_<.25&&linear_>0)linear_=0;pose_.heading=std::remainder(pose_.heading+angular_*dt,6.283185307179586);pose_.x+=std::cos(pose_.heading)*linear_*dt;pose_.y+=std::sin(pose_.heading)*linear_*dt;battery_=std::max(0.0,battery_-dt*(.0002+.001*std::abs(linear_)+.0003*std::abs(angular_)));if(battery_==0)stop();}
void RobotBody::set_obstacle_distance(double m){if(!std::isfinite(m)||m<0)throw std::invalid_argument("invalid obstacle distance");obstacle_=m;}
MemoryStore::MemoryStore(std::size_t c):capacity_(c){if(!c)throw std::invalid_argument("capacity must be positive");}
void MemoryStore::remember(std::string k,std::string v){auto i=std::find_if(data_.begin(),data_.end(),[&](const auto&e){return e.first==k;});if(i!=data_.end())data_.erase(i);data_.emplace_back(std::move(k),std::move(v));while(data_.size()>capacity_)data_.pop_front();}
std::optional<std::string> MemoryStore::recall(const std::string&k)const{for(auto i=data_.rbegin();i!=data_.rend();++i)if(i->first==k)return i->second;return std::nullopt;}
Decision RobotRuntime::act(const Action&a,bool confirmed,const std::string&zone){auto d=security_.authorize(a,confirmed,zone);audit_.push_back(std::string(d.allowed?"ALLOW ":"DENY ")+d.reason);if(audit_.size()>2048)audit_.erase(audit_.begin());if(!d.allowed)return d;switch(a.kind){case Action::Kind::Stop:body_.stop();break;case Action::Kind::Drive:body_.command(a.value,0);break;case Action::Kind::Turn:body_.command(0,a.value);break;case Action::Kind::Speak:memory_.remember("last_speech",a.text);break;}return d;}
void RobotRuntime::tick(double dt){body_.tick(dt);}
}
