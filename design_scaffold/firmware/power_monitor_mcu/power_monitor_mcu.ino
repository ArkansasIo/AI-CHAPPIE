// Telemetry scaffold only; does not replace commercial BMS, fuses or hardware protection.
#include <Arduino.h>
unsigned long lastReport=0;
void setup(){Serial.begin(115200);}
void loop(){if(millis()-lastReport>=250){lastReport=millis();Serial.println("{\"valid\":false,\"reason\":\"MONITOR_NOT_CONFIGURED\"}");}}
