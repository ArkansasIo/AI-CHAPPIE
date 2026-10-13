#include "robot_runtime.hpp"
#include <iostream>
#include <string>
int main(){
 chappie::RobotRuntime robot;
 std::cout<<"AI-CHAPPIE runtime simulator\nCommands: status, drive <value>, turn <value>, stop, tick <seconds>, remember <key> <value>, recall <key>, quit\n";
 std::string cmd;
 while(std::cout<<"> "&&std::cin>>cmd){
  if(cmd=="quit"||cmd=="exit")break;
  if(cmd=="status"){auto p=robot.pose();std::cout<<"pose=("<<p.x<<","<<p.y<<","<<p.heading<<") battery="<<robot.battery()<<"\n";}
  else if(cmd=="stop"){chappie::Action a;a.kind=chappie::Action::Kind::Stop;robot.act(a);std::cout<<"Stopped\n";}
  else if(cmd=="drive"||cmd=="turn"){double v;if(!(std::cin>>v)){std::cin.clear();std::string rest;std::getline(std::cin,rest);std::cout<<"Expected number\n";continue;}chappie::Action a;a.kind=cmd=="drive"?chappie::Action::Kind::Drive:chappie::Action::Kind::Turn;a.value=v;auto d=robot.act(a,true);std::cout<<(d.allowed?"OK: ":"DENIED: ")<<d.reason<<"\n";}
  else if(cmd=="tick"){double dt;if(std::cin>>dt){try{robot.tick(dt);std::cout<<"Tick complete\n";}catch(const std::exception&e){std::cout<<e.what()<<"\n";}}else{std::cin.clear();std::string rest;std::getline(std::cin,rest);}}
  else if(cmd=="remember"){std::string k;std::cin>>k;std::string v;std::getline(std::cin,v);if(!v.empty()&&v.front()==' ')v.erase(0,1);robot.memory().remember(k,v);std::cout<<"Stored\n";}
  else if(cmd=="recall"){std::string k;std::cin>>k;auto v=robot.memory().recall(k);std::cout<<(v?*v:"(not found)")<<"\n";}
  else std::cout<<"Unknown command\n";
 }
 return 0;
}