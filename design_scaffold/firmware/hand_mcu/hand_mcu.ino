// Hand controller scaffold. Finger actuators remain disabled pending force/position limits.
#include <Arduino.h>
#define HAND_ENABLE_PIN 8
void setup(){pinMode(HAND_ENABLE_PIN,OUTPUT);digitalWrite(HAND_ENABLE_PIN,LOW);}
void loop(){digitalWrite(HAND_ENABLE_PIN,LOW);delay(2);}
