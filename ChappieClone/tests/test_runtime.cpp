#include "../src/robot_runtime.hpp"
#include <cassert>
#include <iostream>
int main(){
 using namespace chappie;
 MemoryStore m(2);m.remember("a","1");m.remember("b","2");m.remember("a","3");assert(m.recall("a").value()=="3");m.remember("c","4");assert(!m.recall("b"));
 SecurityManager s;Action a;a.kind=Action::Kind::Drive;a.value=.5;assert(!s.authorize(a,false,"hall").allowed);assert(s.authorize(a,true,"hall").allowed);s.deny_zone("lab");assert(!s.authorize(a,true,"lab").allowed);Action stop;stop.kind=Action::Kind::Stop;assert(s.authorize(stop,false,"lab").allowed);
 OccupancyGrid g(5,5);g.set_blocked(2,1);g.set_blocked(2,2);g.set_blocked(2,3);assert(!g.plan({.5,.5,0},{4.5,.5,0}).empty());
 RobotRuntime r;assert(r.act(a,true).allowed);r.tick(1);assert(r.pose().x>0);r.set_obstacle_distance(.1);double before=r.pose().x;r.tick(1);assert(r.pose().x==before);assert(r.act(stop).allowed);std::cout<<"runtime tests passed\n";
}