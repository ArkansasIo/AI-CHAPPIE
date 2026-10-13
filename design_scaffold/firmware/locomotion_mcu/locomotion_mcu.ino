// Locomotion controller scaffold. No walking or actuator enable implemented.
#include <Arduino.h>
#define DRIVE_ENABLE_PIN 8
void setup(){pinMode(DRIVE_ENABLE_PIN,OUTPUT);digitalWrite(DRIVE_ENABLE_PIN,LOW);}
void loop(){digitalWrite(DRIVE_ENABLE_PIN,LOW);delay(1);}
