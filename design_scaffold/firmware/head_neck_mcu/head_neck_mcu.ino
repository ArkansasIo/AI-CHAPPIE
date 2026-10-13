// Head/neck controller scaffold. Verify hard stops and encoder polarity before use.
#include <Arduino.h>
#define NECK_ENABLE_PIN 8
void setup(){pinMode(NECK_ENABLE_PIN,OUTPUT);digitalWrite(NECK_ENABLE_PIN,LOW);}
void loop(){digitalWrite(NECK_ENABLE_PIN,LOW);delay(2);}
