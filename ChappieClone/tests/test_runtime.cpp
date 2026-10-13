#include "../src/runtime/robot_runtime.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 using namespace chappie;
 MemoryStore m(2);m.remember("a","1");m.remember("b","2");m.remember("a","3");assert(m.recall("a").value()=="3");m.remember("c","4");assert(!m.recall("b"));
 SecurityManager sec(AutonomyLevel::Assisted);Action drive;drive.kind=ActionKind::Drive;drive.a=.5;assert(!sec.authorize(drive,false,"hall").allowed);assert(sec.authorize(drive,true,"hall").allowed);sec.deny_zone("lab");assert(!sec.authorize(drive,true,"lab").allowed);Action stop;stop.kind=ActionKind::Stop;assert(sec.authorize(stop,false,"lab").allowed);
 OccupancyGrid grid(5,5,1);grid.set_blocked(2,1);grid.set_blocked(2,2);grid.set_blocked(2,3);auto path=grid.plan({.5,.5},{4.5,.5});assert(!path.empty());assert(path.front().x==.5&&path.back().x==4.5);
 RobotRuntime robot(AutonomyLevel::Assisted);assert(robot.act(drive,true).allowed);robot.tick(1);assert(robot.sense().sequence==1);assert(robot.sense().pose.position.x>0);robot.set_obstacle_distance(.1);robot.tick(1);assert(robot.body().velocity().linear==0);Action stop2;stop2.kind=ActionKind::Stop;assert(robot.act(stop2).allowed);std::cout<<"All runtime tests passed\n";
}
