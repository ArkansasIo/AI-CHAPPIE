#include "runtime/robot_runtime.hpp"
#include <iostream>
#include <string>

int main(int argc,char** argv){
  chappie::RobotRuntime robot(chappie::AutonomyLevel::Assisted);
  std::cout<<"AI-CHAPPIE runtime simulator\nCommands: status, drive <normalized -1..1>, turn <normalized -1..1>, stop, tick <seconds>, remember <key> <value>, recall <key>, quit\n";
  std::string command;
  while(std::cout<<"> "&&std::cin>>command){
    if(command=="quit"||command=="exit")break;
    if(command=="status"){auto s=robot.sense();auto p=s.pose;std::cout<<"tick="<<s.sequence<<" pose=("<<p.position.x<<","<<p.position.y<<","<<p.heading<<") battery="<<s.battery_fraction<<" stopped="<<robot.body().stopped()<<"\n";}
    else if(command=="drive"||command=="turn"){double v;if(!(std::cin>>v)){std::cin.clear();std::string rest;std::getline(std::cin,rest);std::cout<<"Expected numeric value\n";continue;}chappie::Action a;a.kind=command=="drive"?chappie::ActionKind::Drive:chappie::ActionKind::Turn;a.a=v;auto d=robot.act(a,true);std::cout<<(d.allowed?"OK: ":"DENIED: ")<<d.reason<<"\n";}
    else if(command=="stop"){chappie::Action a;a.kind=chappie::ActionKind::Stop;robot.act(a);std::cout<<"Stopped\n";}
    else if(command=="tick"){double dt;if(std::cin>>dt){try{robot.tick(dt);std::cout<<"Tick complete\n";}catch(const std::exception& e){std::cout<<e.what()<<"\n";}}else{std::cin.clear();std::string rest;std::getline(std::cin,rest);}}
    else if(command=="remember"){std::string key;std::cin>>key;std::string value;std::getline(std::cin,value);if(!value.empty()&&value.front()==' ')value.erase(0,1);robot.memory().remember(key,value);std::cout<<"Stored\n";}
    else if(command=="recall"){std::string key;std::cin>>key;auto v=robot.memory().recall(key);std::cout<<(v?*v:"(not found)")<<"\n";}
    else std::cout<<"Unknown command\n";
  }
  return 0;
}
