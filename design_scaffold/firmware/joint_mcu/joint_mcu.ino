// Joint controller scaffold; outputs disabled by default.
#include <Arduino.h>
#define JOINT_ENABLE_PIN 8
void setup(){pinMode(JOINT_ENABLE_PIN,OUTPUT);digitalWrite(JOINT_ENABLE_PIN,LOW);}
void loop(){// Add vendor drive API only after actuator selection and review.
  digitalWrite(JOINT_ENABLE_PIN,LOW);delay(1);}
