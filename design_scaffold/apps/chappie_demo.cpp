#include "brain/BrainOS.hpp"
#include "boot/BootManager.hpp"
#include "hardware/SimulatedHardware.hpp"
#include "safety/LawEngine.hpp"
#include "safety/SafetySupervisor.hpp"
#include "model/RobotModel.hpp"
#include <iostream>
int main(){
  chappie::BrainOS brain; chappie::HealthSnapshot h{true,true,true,true,true,true}; brain.updateHealth(h);
  chappie::CommandEnvelope c; c.command_id="demo-1";c.action="status";c.requester_id="local-demo";c.permission="read_status";c.authenticated=true;c.authorized=true;c.risk=chappie::Risk::Low;
  const auto r=brain.request(c); std::cout<<"AI-CHAPPIE demo decision: "<<(r.decision==chappie::Decision::Allow?"CMD-ALLOW":"CMD-DENY")<<" — "<<r.reason<<"\n";
  return r.decision==chappie::Decision::Allow?0:1;
}
