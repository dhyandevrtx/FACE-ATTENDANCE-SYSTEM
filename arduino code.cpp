/**
 * Arduino Code for Face Recognition Attendance System Feedback
 * 
 * - Receives 'P', 'A', 'I' commands from Python via Serial.
 * - Controls LEDs:
 *     - Pin 8: Green (Present)
 *     - Pin 6: Red (Absent/Unknown) 
 *     - Pin 10: Yellow/Blue (Idle)
 * - Controls a SERVO MOTOR on pin 9.
 * - Controls a BUZZER on pin 13:
 *     - ON when receiving 'P' or 'A' (Active Tracking)
 *     - OFF when receiving 'I' (Idle/Stopped)
 * - Periodically reads sensors (LDR A0, Failure Input 7).
 * - Sends sensor data back to Python.
 */

#include <Servo.h> // Include the Servo library

// --- Pin Definitions ---
const int presentLedPin = 8;  // Green LED for Present/Recognized
const int absentLedPin = 6;   // Red LED (Absent/Unknown)
const int servoPin = 9;       // Digital PWM pin for the Servo signal wire
const int idleLedPin = 10;    // Yellow/Blue LED for Idle/Waiting/Stopped
const int buzzerPin = 13;     // Buzzer Pin (often built-in LED too)

// --- Sensor Pin Definitions ---
const int ldrPin = A0;        // Analog pin for Light Dependent Resistor (LDR)
const int failureInputPin = 7;// Digital pin to simulate failure (GND = fail)

// --- Servo Control ---
Servo flagServo;             // Create a servo object
const int PRESENT_ANGLE = 10;  // <<< ADJUST Angle for "Present" (e.g., Green Flag visible)
const int ABSENT_ANGLE = 170; // <<< ADJUST Angle for "Absent/Unknown" (e.g., Red Flag visible)
const int IDLE_ANGLE = 90;    // <<< ADJUST Angle for "Idle" (e.g., middle position)

// --- Timing for Sensor Readings ---
unsigned long lastSensorReadTime = 0;
const unsigned long sensorReadInterval = 5000; // Read sensors every 5 seconds

void setup() {
  Serial.begin(9600); // Match Python's BAUD_RATE

  // Initialize Output pins
  pinMode(presentLedPin, OUTPUT);
  pinMode(absentLedPin, OUTPUT);  
  pinMode(idleLedPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);     

  // Initialize Input pins
  pinMode(failureInputPin, INPUT_PULLUP); // Use internal pull-up resistor

  // Attach servo
  flagServo.attach(servoPin);
  Serial.println("Arduino: Servo attached to pin " + String(servoPin));

  // Initial state: LEDs off (except Idle), Buzzer OFF, Servo to Idle
  digitalWrite(presentLedPin, LOW);
  digitalWrite(absentLedPin, LOW);   
  digitalWrite(idleLedPin, HIGH);    
  digitalWrite(buzzerPin, LOW);      
  flagServo.write(IDLE_ANGLE);       
  delay(500); // Allow servo to reach position

  while (!Serial); // Optional: wait for serial port to connect
  Serial.println("Arduino: Initialized and Ready (Servo, LEDs, Buzzer, Sensors).");
}

void loop() {

  // --- Task 1: Check for and Process Incoming Commands from Python ---
  if (Serial.available() > 0) {
    char receivedCode = Serial.read();

    switch (receivedCode) {
      case 'P': // Present/Recognized (Active Tracking)
        digitalWrite(presentLedPin, HIGH); // Green ON
        digitalWrite(absentLedPin, LOW);   // Red OFF  
        digitalWrite(idleLedPin, LOW);     // Idle OFF
        digitalWrite(buzzerPin, HIGH);     // Buzzer ON
        flagServo.write(PRESENT_ANGLE);   
        break;

      case 'A': // Absent/Unknown (Active Tracking)
        digitalWrite(presentLedPin, LOW); 
        digitalWrite(absentLedPin, HIGH);  // Red ON   
        digitalWrite(idleLedPin, LOW);     // Idle OFF
        digitalWrite(buzzerPin, HIGH);     // Buzzer ON
        flagServo.write(ABSENT_ANGLE);    
        break;

      case 'I': // Idle/Waiting/Stopped
        digitalWrite(presentLedPin, LOW);
        digitalWrite(absentLedPin, LOW);   // Red OFF  
        digitalWrite(idleLedPin, HIGH);    // Idle LED ON
        digitalWrite(buzzerPin, LOW);      // Buzzer OFF
        flagServo.write(IDLE_ANGLE);       
        break;

      default: // Unknown command (Treat as Idle)
        digitalWrite(presentLedPin, LOW);
        digitalWrite(absentLedPin, LOW);   // Red OFF  
        digitalWrite(idleLedPin, HIGH);    // Idle LED ON
        digitalWrite(buzzerPin, LOW);      // Buzzer OFF
        flagServo.write(IDLE_ANGLE);       
        break;
    }
  } // End of checking incoming Serial data


  // --- Task 2: Periodically Read Sensors and Send Data Back to Python ---
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorReadTime >= sensorReadInterval) {
    lastSensorReadTime = currentMillis; // Update the timer

    // Read LDR
    int ldrValue = analogRead(ldrPin);
    Serial.print("LDR:");
    Serial.println(ldrValue);

    // Check failure input (LOW = Failed because using INPUT_PULLUP)
    bool isSensorFailed = (digitalRead(failureInputPin) == LOW);
    if (isSensorFailed) {
      Serial.println("Status:SensorFailure");
    } else {
      Serial.println("Status:OK");
    }
  } // End periodic sensor reading

  // No delay() in the main loop - keeps Arduino responsive
  
} // End of loop()
