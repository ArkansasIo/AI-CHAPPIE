// Scaffold only. Not safety-certified. Physical E-stop must use a reviewed hardware safety circuit.
#include <Arduino.h>
#define ESTOP_PIN 2
#define HEARTBEAT_PIN 3
#define ENABLE_OUT_PIN 8
const unsigned long HEARTBEAT_TIMEOUT_MS=250;
unsigned long lastHeartbeat=0;
bool faultLatched=true;
void setup(){pinMode(ESTOP_PIN,INPUT_PULLUP);pinMode(HEARTBEAT_PIN,INPUT_PULLUP);pinMode(ENABLE_OUT_PIN,OUTPUT);digitalWrite(ENABLE_OUT_PIN,LOW);lastHeartbeat=millis();}
void loop(){
  if(digitalRead(HEARTBEAT_PIN)==HIGH) lastHeartbeat=millis();
  if(digitalRead(ESTOP_PIN)==LOW || millis()-lastHeartbeat>HEARTBEAT_TIMEOUT_MS) faultLatched=true;
  // Intentionally never enables actuators; integrate a reviewed safety device.
  digitalWrite(ENABLE_OUT_PIN,LOW); delay(1);
}
