#include <Wire.h>
#include <Zumo32U4.h>

Zumo32U4Motors motors;
Zumo32U4Buzzer buzzer;
Zumo32U4ButtonC buttonC;
Zumo32U4ProximitySensors proxSensors;

LSM303 compass;

#define BUFFER_SIZE 80

char inputBuffer[BUFFER_SIZE];
int index = 0;

void setup()
{
  Serial.begin(9600);

  // Initialize the LSM303D accelerometer.
  compass.init();

  // Initialize proximity sensors
  proxSensors.initThreeSensors();

  Serial.println("Initialization done.");

  delay(1000);
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

bool listenCommand() {
  bool cmdFound = false;
  while (Serial.available() > 0) {
    char cmdBuffer = Serial.read();
    if (cmdBuffer == '\n') {
      inputBuffer[index] = '\0';
      cmdFound = index > 0;
      index = 0;
      break;
    } else if (index < BUFFER_SIZE && !cmdFound) {
        inputBuffer[index++] = cmdBuffer;
      }
  }
  return cmdFound;
}

void beep(byte n) {
  for (byte i = 0;i < n; i++) {
    buzzer.playFrequency(440, 100, 10);
    delay(500);
  }
}

void listenEvents() {
  if (listenCommand()) {
    switch (inputBuffer[0]) {
      case 'W': {
        int leftWheel;
        int rightWheel;
        sscanf(inputBuffer, "W, %d, %d", &leftWheel, &rightWheel);
        motors.setSpeeds(leftWheel, rightWheel);
      }
     case 'R': {
        motors.setSpeeds(0, 0);
     }
    }
  }
}

void loop()
{ 
  // sendData();
  listenEvents();
}
