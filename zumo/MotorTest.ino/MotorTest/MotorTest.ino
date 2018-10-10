#include <Wire.h>
#include <Zumo32U4.h>

Zumo32U4Motors motors;
Zumo32U4Buzzer buzzer;
Zumo32U4ButtonC buttonC;
Zumo32U4ProximitySensors proxSensors;

LSM303 compass;

#define BUFFER_SIZE 80
#define SPEED           200 // Maximum motor speed when going straight; variable speed when turning
#define CALIBRATION_SAMPLES 70  // Number of compass readings to take when calibrating
#define CRB_REG_M_2_5GAUSS 0x60 // CRB_REG_M value for magnetometer +/-2.5 gauss full scale
#define CRA_REG_M_220HZ    0x1C // CRA_REG_M value for magnetometer 220 Hz update rate

char inputBuffer[BUFFER_SIZE];
int index = 0;

// The highest possible magnetic value to read in any direction is 2047
// The lowest possible magnetic value to read in any direction is -2047
LSM303::vector<int16_t> running_min = {32767, 32767, 32767}, running_max = {-32767, -32767, -32767};
unsigned char compassIndex;

void setup()
{
  Serial.begin(9600);  // Begin the serial monitor at 9600bps

  Serial.println("starting setup");

  // Initiate the Wire library and join the I2C bus as a master
  Wire.begin();

  // Initialize the LSM303D accelerometer.
  compass.init();

  // Enables accelerometer and magnetometer
  compass.enableDefault();

  compass.writeReg(LSM303::CRB_REG_M, CRB_REG_M_2_5GAUSS); // +/- 2.5 gauss sensitivity to hopefully avoid overflow problems
  compass.writeReg(LSM303::CRA_REG_M, CRA_REG_M_220HZ);    // 220 Hz compass update rate
  
  
  // Initialize proximity sensors
  proxSensors.initThreeSensors();

  Serial.println("Initialization done.");

  delay(1000);
}

void calibrateCompass() {
  Serial.println("starting calibration");

  // To calibrate the magnetometer, the Zumo spins to find the max/min
  // magnetic vectors. This information is used to correct for offsets
  // in the magnetometer data.
  motors.setLeftSpeed(SPEED);
  motors.setRightSpeed(-SPEED);

  for(index = 0; index < CALIBRATION_SAMPLES; index ++)
  {
      // Take a reading of the magnetic vector and store it in compass.m
      compass.read();
    
      running_min.x = min(running_min.x, compass.m.x);
      running_min.y = min(running_min.y, compass.m.y);
      running_min.z = min(running_min.z, compass.m.z);
    
      running_max.x = max(running_max.x, compass.m.x);
      running_max.y = max(running_max.y, compass.m.y);
      running_max.z = max(running_max.z, compass.m.z);
        
      delay(50);
  }
  
  motors.setLeftSpeed(0);
  motors.setRightSpeed(0);

  // Set calibrated values to compass.m_max and compass.m_min
  compass.m_max.x = running_max.x;
  compass.m_max.y = running_max.y;
  compass.m_max.z = running_max.z;
  compass.m_min.x = running_min.x;
  compass.m_min.y = running_min.y;
  compass.m_min.z = running_min.z;

  snprintf(inputBuffer, BUFFER_SIZE, "min: { %d, %d, %d } max: { %d. %d, %d }",
  running_min.x, running_min.y, running_min.z,
  running_max.x, running_max.y, running_max.z);

  Serial.println(inputBuffer);
}

void readProxSensors() {
  proxSensors.read();
  compass.read();
}

void sendData() {
  static char buffer[80];
  sprintf(buffer, "S %d %d %d %d %d %d",
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
    buzzer.playFrequency(440, 50, 10);
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
        break;
      }
     case 'R': {
        motors.setSpeeds(0, 0);
        break;
     }
     case 'N': {
        beep(1);
        break;
     }
     case 'F': {
        motors.setSpeeds(SPEED, SPEED);
        break;
     }
     case 'B': {
        motors.setSpeeds(-SPEED, -SPEED);
        break;
     }
     case 'S': {
      readProxSensors();
      sendData();
      break;
     }
    }
  }
}

void loop()
{ 
  listenEvents();
}
