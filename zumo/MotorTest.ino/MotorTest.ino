#include <Wire.h>
#include <Zumo32U4.h>

Zumo32U4Motors motors;
Zumo32U4Buzzer buzzer;
Zumo32U4ButtonC buttonC;
Zumo32U4ProximitySensors proxSensors;

LSM303 compass;


void setup()
{
  Serial.begin(9600);

  // Initialize the LSM303D accelerometer.
  compass.init();

  // Initialize proximity sensors
  proxSensors.initThreeSensors();

  Serial.println("Initialization done.");
}

void readProxSensors() {
  proxSensors.read();
}

void sendData() {
  static char buffer[80];
  sprintf(buffer, "%d %d %d %d %d %d",
    proxSensors.countsLeftWithLeftLeds(),
    proxSensors.countsLeftWithRightLeds(),
    proxSensors.countsFrontWithLeftLeds(),
    proxSensors.countsFrontWithRightLeds(),
    proxSensors.countsRightWithLeftLeds(),
    proxSensors.countsRightWithRightLeds()
  );
  Serial.print(buffer);
  Serial.println("Data sent");
  delay(1000);
}

void loop()
{
  sendData();
  char buff = Serial.read();
  Serial.println(buff);
}
