// Sensor hub scaffold; no sensor data is marked valid until drivers and freshness checks are implemented.
#include <Arduino.h>
void setup(){Serial.begin(115200);}
void loop(){static unsigned long last=0;if(millis()-last>=100){last=millis();Serial.println("{\"valid\":false,\"reason\":\"SENSORS_NOT_CONFIGURED\"}");}}
