#include "brain/BrainOS.hpp"
#include "model/RobotModel.hpp"
#include "safety/LawEngine.hpp"
#include "safety/SafetySupervisor.hpp"
#include "hardware/SimulatedHardware.hpp"
#include <cassert>
#include <limits>
int main(){
  chappie::LawEngine law; chappie::CommandEnvelope c; assert(law.evaluate(c).decision==chappie::Decision::Deny);
  c.command_id="1";c.action="status";c.requester_id="test";c.permission="read_status";c.authenticated=true;c.authorized=true;c.risk=chappie::Risk::Low;assert(law.evaluate(c).decision==chappie::Decision::Allow);
  c.ambiguous=true;assert(law.evaluate(c).decision==chappie::Decision::RequireConfirmation);
  chappie::RobotModel model;assert(model.joints().size()==28);assert(model.validTarget(0,0.0f,0.1f));assert(!model.validTarget(0,std::numeric_limits<float>::quiet_NaN(),0.1f));
  chappie::SafetySupervisor safety; chappie::MotionRequest req{200,0,0.0f,0.1f};chappie::SafetyInputs inputs{100,true,true,true,true};assert(safety.evaluate(req,inputs).allow);inputs.estop_clear=false;assert(!safety.evaluate(req,inputs).allow);
  chappie::SimulatedHardware hardware;assert(hardware.state()==chappie::HardwareState::SafeIdle);assert(hardware.move({0.05f,0.0f,50})==chappie::HardwareResult::Ok);assert(hardware.stop()==chappie::HardwareResult::Ok);
}
