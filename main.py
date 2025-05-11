import tkinter as tk
from tkinter import ttk, font as tkFont, messagebox as mess
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import serial # Import pyserial

# --- Configuration ---
# !!! IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S PORT !!!
# Examples: 'COM3' (Windows), '/dev/ttyACM0' or '/dev/ttyUSB0' (Linux), '/dev/cu.usbmodemXXXX' (macOS)
ARDUINO_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600          # Must match Arduino's Serial.begin() rate
CAMERA_INDEX = 0          # Change if you have multiple cameras
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml' # Assumed in same folder
CONFIDENCE_THRESHOLD = 50 # Lower value means more strict matching (LBPH specific)

# --- File Paths ---
path = os.path.dirname(os.path.abspath(__file__))
haar_cascade_file = os.path.join(path, HAARCASCADE_PATH)
training_image_folder = os.path.join(path, "TrainingImage")
training_label_folder = os.path.join(path, "TrainingImageLabel")
trainer_file = os.path.join(training_label_folder, "Trainner.yml")
student_details_folder = os.path.join(path, "StudentDetails")
student_details_file = os.path.join(student_details_folder, "StudentDetails.csv")
attendance_folder = os.path.join(path, "Attendance")

# Create directories if they don't exist
for folder in [training_image_folder, training_label_folder, student_details_folder, attendance_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Arduino Communication Setup ---
arduino = None # Initialize arduino variable
arduino_status = "Arduino: N/A" # Default status message for GUI

try:
    print(f"Attempting to connect to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud...")
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1) # timeout=1 second for reads
    time.sleep(2) # Give Arduino time to reset after connection established
    print(f"Arduino connection established on {ARDUINO_PORT}.")
    # Optional: Send an initial 'Idle' state code (e.g., 'I') to Arduino
    try:
        arduino.write(b'I') # Send 'I' as bytes
        print("Sent initial 'I' (Idle) state to Arduino.")
        arduino_status = f"Arduino: OK ({ARDUINO_PORT})"
    except Exception as write_err:
         print(f"Warning: Could not send initial 'I' state to Arduino: {write_err}")
         arduino_status = f"Arduino: Connected but initial send failed ({ARDUINO_PORT})"

except serial.SerialException as e:
    print(f"--- Arduino Connection Error ---")
    print(f"Error connecting to Arduino on {ARDUINO_PORT}: {e}")
    print("Hardware feedback will be disabled.")
    mess.showerror("Arduino Error", f"Could not connect to Arduino on {ARDUINO_PORT}.\n{e}\n\nPlease check:\n1. Is it plugged in?\n2. Is the correct port selected?\n3. Is the port free (Serial Monitor closed)?\n\nHardware feedback will be disabled.")
    arduino = None # Ensure it's None if connection failed
    arduino_status = "Arduino: Connection Failed"
except Exception as e:
    # Catch other potential errors during setup (less common)
    print(f"--- An unexpected error occurred during Arduino connection ---")
    print(f"{e}")
    mess.showerror("Arduino Error", f"An unexpected error occurred during Arduino connection.\n{e}\n\nHardware feedback will be disabled.")
    arduino = None
    arduino_status = "Arduino: Error"


# --- Function Definitions ---

def check_haarcascadefile():
    """Checks if the Haar Cascade file exists."""
    exists = os.path.isfile(haar_cascade_file)
    if exists:
        # print("Haar cascade file found:", HAARCASCADE_PATH) # Less verbose
        pass
    else:
        mess.showerror('Error', f'Critical Error: Haar Cascade file not found!\nExpected at: {haar_cascade_file}')
        window.destroy() # Stop if cascade is missing
    return exists

def is_number(s):
    """Checks if a string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False # Simpler check is usually sufficient

def update_clock():
    """Updates the date and time labels periodically."""
    now = time.strftime("%d-%b-%Y %H:%M:%S")
    date_label.config(text=now.split()[0])
    time_label.config(text=now.split()[1])
    window.after(1000, update_clock) # Schedule next update

def clear_id_name():
    """Clears the ID and Name entry fields."""
    txt_id.delete(0, 'end')
    txt_name.delete(0, 'end')
    message_label.config(text="Registration fields cleared.")

def clear_treeview():
    """Clears the attendance log display (Treeview)."""
    count = 0
    for item in tv.get_children():
        tv.delete(item)
        count += 1
    res = f"Attendance log display cleared ({count} items removed)."
    message_label.config(text=res)

def display_attendance_log():
    """Loads and displays today's attendance log in the Treeview."""
    clear_treeview() # Clear existing entries first
    today_date = datetime.date.today().strftime("%d-%m-%Y") # Consistent format
    attendance_file_path = os.path.join(attendance_folder, f"Attendance_{today_date}.csv")

    if not os.path.isfile(attendance_file_path):
        res = f"No attendance log found for today ({today_date})."
        message_label.config(text=res)
        return

    try:
        count = 0
        with open(attendance_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None) # Read header, handle empty file
            if header is None:
                message_label.config(text=f"Attendance file for {today_date} is empty.")
                return

            for row in reader:
                if len(row) >= 4: # Ensure row has enough columns (ID, Name, Date, Time)
                    tv.insert('', 'end', values=(row[0], row[1], row[2], row[3]))
                    count += 1
                # else: print(f"Skipping malformed row: {row}") # Optional: Log bad rows
            res = f"Displayed {count} attendance records for {today_date}."
            message_label.config(text=res)

    except FileNotFoundError:
         res = f"Attendance file not found for today ({today_date})."
         message_label.config(text=res)
    except Exception as e:
        mess.showerror("CSV Error", f"Error reading attendance file:\n{attendance_file_path}\n\nError: {e}")
        message_label.config(text="Error loading attendance log.")

def send_to_arduino(code_byte):
    """Sends a single byte code to Arduino if connected."""
    global arduino, arduino_status # Allow modification of global status
    if arduino:
        try:
            arduino.write(code_byte)
            # print(f"Sent '{code_byte.decode()}' to Arduino") # Debug: uncomment to verify sending
        except serial.SerialTimeoutException:
            print(f"Warning: Timeout writing to Arduino ({code_byte.decode()}).")
            # Optionally try to reconnect or just notify
        except serial.SerialException as e:
            print(f"--- Arduino Send Error ---")
            print(f"Error sending data '{code_byte.decode()}' to Arduino: {e}")
            mess.showerror("Arduino Error", f"Lost connection while sending data.\nPlease check Arduino.\n\nError: {e}")
            arduino_status = "Arduino: Disconnected"
            arduino_status_label.config(text=arduino_status)
            try:
                arduino.close()
            except: pass
            arduino = None # Stop further attempts
        except Exception as e:
            print(f"--- Unexpected Arduino Send Error ---")
            print(f"Error: {e}")


def take_images():
    """Captures images for a new student registration."""
    if not check_haarcascadefile(): return
    student_id = txt_id.get().strip()
    student_name = txt_name.get().strip()

    # --- Input Validation ---
    if not student_id or not student_name:
        mess.showerror('Input Error', 'Student ID and Name are required.')
        return
    if not is_number(student_id):
        mess.showerror('Input Error', 'Student ID must be a numeric value.')
        return
    # Simple check for name validity (can be improved)
    if not all(c.isalpha() or c.isspace() for c in student_name):
         mess.showerror('Input Error', 'Student Name should contain only letters and spaces.')
         return

    cam = None
    try:
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            raise IOError(f"Cannot open webcam with index {CAMERA_INDEX}")

        face_detector = cv2.CascadeClassifier(haar_cascade_file)
        student_id_int = int(student_id) # Use integer ID for comparison

        # --- Check if ID already exists ---
        existing_ids = set()
        if os.path.isfile(student_details_file):
            try:
                df_details = pd.read_csv(student_details_file)
                if not df_details.empty:
                    existing_ids.update(df_details['ID'].astype(int).tolist())
            except pd.errors.EmptyDataError:
                pass # File exists but is empty
            except Exception as e:
                 mess.showwarning("CSV Warning", f"Could not read existing student details: {e}")

        if student_id_int in existing_ids:
             overwrite = mess.askyesno("ID Exists", f"Student ID {student_id} already exists.\nDo you want to overwrite the images (details remain)?")
             if not overwrite:
                 message_label.config(text="Image capture cancelled.")
                 return
             else:
                 print(f"Proceeding to overwrite images for existing ID: {student_id}")
                 # Optional: delete old images for this ID first
        else:
             # Add new student details only if ID is new
             row = [student_id, student_name]
             try:
                  # Open in append mode, create header if file is new/empty
                  file_exists = os.path.isfile(student_details_file)
                  with open(student_details_file, 'a+', newline='') as csvFile:
                       writer = csv.writer(csvFile)
                       # Write header if file is new or empty
                       if not file_exists or os.path.getsize(student_details_file) == 0:
                            writer.writerow(['ID', 'Name'])
                       writer.writerow(row)
                  print(f"New student details saved: ID={student_id}, Name={student_name}")
             except Exception as e:
                  print(f"Error saving student details: {e}")
                  mess.showerror("Save Error", f"Could not save student details to CSV:\n{e}")
                  # Decide if you want to proceed with image capture even if details save failed
                  # return

        # --- Image Capture Loop ---
        message_label.config(text=f"Taking images for {student_name} (ID: {student_id})... Look at the camera.")
        window.update_idletasks() # Force GUI update

        img_count = 0
        max_images = 100 # Number of images to capture
        while img_count < max_images:
            ret, img = cam.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.")
                time.sleep(0.1) # Wait a bit before retrying
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adjust detection parameters if needed (more sensitive: lower scaleFactor, minNeighbors)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img_count += 1

                # Save the captured face region (grayscale)
                face_img = gray[y:y + h, x:x + w]
                img_path = os.path.join(training_image_folder, f"{student_name}.{student_id}.{img_count}.jpg")

                # Create folder if it doesn't exist (should be handled earlier, but good safeguard)
                if not os.path.exists(training_image_folder): os.makedirs(training_image_folder)

                cv2.imwrite(img_path, face_img)

                # Display progress on the image/window title
                progress_text = f"Capturing: {img_count}/{max_images}"
                cv2.putText(img, progress_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow('Taking Images - Press "q" to Cancel', img)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(50) & 0xFF == ord('q'): # Check key press every 50ms
                message_label.config(text="Image capture cancelled by user.")
                print("Image capture cancelled by user.")
                img_count = max_images + 1 # Ensure loop terminates, but indicate cancellation
                break # Exit the while loop

        # --- Post-Capture Cleanup & Messaging ---
        if img_count == max_images: # Completed successfully
             message_label.config(text=f"Successfully captured {max_images} images for {student_name}.")
             print(f"Successfully captured {max_images} images for ID: {student_id}.")
             mess.showinfo("Capture Complete", f"{max_images} images captured for\nID: {student_id}\nName: {student_name}")
        elif img_count < max_images: # Loop finished unexpectedly (e.g., camera error after start)
             message_label.config(text=f"Image capture stopped. Only {img_count} images saved.")
             print(f"Warning: Image capture stopped prematurely. Only {img_count} images saved for ID: {student_id}.")
        # else: # User cancelled (img_count > max_images)
             # Message already set inside the loop

    except IOError as e:
         print(f"Camera Error: {e}")
         mess.showerror("Camera Error", f"Failed to access the camera (Index {CAMERA_INDEX}).\nIs it connected and not in use?\nError: {e}")
         message_label.config(text="Camera error occurred.")
    except Exception as e:
         print(f"An error occurred during image capture: {e}")
         mess.showerror("Error", f"An unexpected error occurred during image capture:\n{e}")
         message_label.config(text="Image capture failed.")
    finally:
        if cam and cam.isOpened():
            cam.release()
        cv2.destroyAllWindows()
        print("Image capture resources released.")


def train_images():
    """Trains the LBPH face recognizer using captured images."""
    if not check_haarcascadefile(): return # Should already be checked, but good practice

    image_paths = [os.path.join(training_image_folder, f) for f in os.listdir(training_image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))] # Filter for image files
    faces = []
    ids = []

    if not image_paths:
        mess.showerror("Training Error", "No images found in the 'TrainingImage' folder.\nPlease capture images first.")
        message_label.config(text="Training failed: No images found.")
        return

    message_label.config(text="Starting image training... This may take a moment.")
    window.update_idletasks() # Show message immediately

    total_images = 0
    skipped_files = 0
    processed_ids = set()

    for imagePath in image_paths:
        try:
            # Open image using PIL, convert to grayscale
            pil_image = Image.open(imagePath).convert('L')
            # Convert PIL image to NumPy array
            image_np = np.array(pil_image, dtype='uint8')

            # Extract the ID from the filename (e.g., "Name.ID.Count.jpg")
            filename = os.path.split(imagePath)[-1]
            parts = filename.split(".")
            if len(parts) < 3:
                 print(f"Warning: Skipping file with unexpected format: {filename}")
                 skipped_files += 1
                 continue

            img_id = int(parts[1]) # The second part should be the ID

            # --- Optional: Basic face detection validation during training ---
            # detector = cv2.CascadeClassifier(haar_cascade_file)
            # detected_faces = detector.detectMultiScale(image_np)
            # if len(detected_faces) != 1: # Expect exactly one face per training image
            #     print(f"Warning: Skipping {filename} - Expected 1 face, found {len(detected_faces)}")
            #     skipped_files += 1
            #     continue
            # ---------------------------------------------------------------

            faces.append(image_np)
            ids.append(img_id)
            processed_ids.add(img_id)
            total_images += 1

        except ValueError:
             print(f"Warning: Skipping {filename} - Could not parse ID as integer.")
             skipped_files += 1
        except Exception as e:
            print(f"Warning: Skipping file {imagePath} due to error: {e}")
            skipped_files += 1

    if not faces or not ids:
        mess.showerror("Training Error", "No valid images could be processed for training.")
        message_label.config(text="Training failed: No valid images.")
        return

    ids = np.array(ids) # Convert list of IDs to NumPy array

    # --- Train the Recognizer ---
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create() # Create LBPH recognizer
        print("Training LBPH recognizer...")
        recognizer.train(faces, ids) # Train the model

        # Ensure the label folder exists
        if not os.path.exists(training_label_folder):
            os.makedirs(training_label_folder)

        # Save the trained model
        recognizer.write(trainer_file)

        num_students = len(processed_ids)
        print(f"Recognizer trained successfully on {total_images} images for {num_students} unique students.")
        if skipped_files > 0:
             print(f"Skipped {skipped_files} files during training.")

        mess.showinfo("Training Complete", f"Recognizer trained successfully!\n\nImages Processed: {total_images}\nUnique Students: {num_students}\nSkipped Files: {skipped_files}\n\nModel saved to:\n{trainer_file}")
        message_label.config(text="Training complete.")

    except cv2.error as cv_err: # Catch OpenCV specific errors
         print(f"OpenCV Error during training/saving: {cv_err}")
         mess.showerror("Training Error (OpenCV)", f"An OpenCV error occurred:\n{cv_err}\n\nEnsure images are valid and you have sufficient permissions.")
         message_label.config(text="Training failed (OpenCV error).")
    except Exception as e:
        print(f"Error during training or saving model: {e}")
        mess.showerror("Training Error", f"An unexpected error occurred during training or saving the model:\n{e}")
        message_label.config(text="Training failed.")


def track_images():
    """Tracks faces via webcam, recognizes them, records attendance, and sends signals to Arduino."""
    global arduino # Access the global arduino object

    # --- Pre-checks ---
    if not check_haarcascadefile(): return
    if not os.path.isfile(trainer_file):
        mess.showerror("Tracking Error", f"Trainer file not found!\n({trainer_file})\nPlease train images first.")
        message_label.config(text="Error: Trainer file missing.")
        return
    if not os.path.isfile(student_details_file):
        mess.showerror("Tracking Error", f"Student details file not found!\n({student_details_file})\nPlease register students first.")
        message_label.config(text="Error: Student details missing.")
        return

    cam = None
    try:
        # --- Load Recognizer and Data ---
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_file)
        face_cascade = cv2.CascadeClassifier(haar_cascade_file)

        try:
            df = pd.read_csv(student_details_file)
            if df.empty:
                raise pd.errors.EmptyDataError # Treat empty file as error for tracking
            # Ensure ID column is string for consistent lookup later
            df['ID'] = df['ID'].astype(str)
        except pd.errors.EmptyDataError:
            mess.showerror("Tracking Error", f"Student details file is empty!\n({student_details_file})\nPlease register students.")
            message_label.config(text="Error: Student details empty.")
            return
        except FileNotFoundError: # Should have been caught earlier, but safeguard
             mess.showerror("Tracking Error", f"Student details file not found!\n({student_details_file})")
             return
        except Exception as e:
             mess.showerror("CSV Error", f"Error reading student details CSV:\n{e}")
             return

        print("Recognizer, cascade, and student details loaded.")

        # --- Setup Camera ---
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            raise IOError(f"Cannot open webcam with index {CAMERA_INDEX}")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Optional: Set resolution
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Prepare Attendance File ---
        today_date = datetime.date.today().strftime("%d-%m-%Y")
        attendance_file_path = os.path.join(attendance_folder, f"Attendance_{today_date}.csv")
        attendance_header = ['ID', 'Name', 'Date', 'Time']
        file_exists = os.path.isfile(attendance_file_path)

        # Read existing IDs logged *today* to avoid duplicate entries in this session
        logged_ids_today = set()
        if file_exists:
            try:
                # Only read IDs if file is not empty
                if os.path.getsize(attendance_file_path) > 0:
                    attendance_df = pd.read_csv(attendance_file_path)
                    if not attendance_df.empty and 'ID' in attendance_df.columns:
                         logged_ids_today.update(attendance_df['ID'].astype(str).tolist())
            except pd.errors.EmptyDataError:
                print(f"Attendance file {attendance_file_path} is empty.") # Normal case if first run of day
            except Exception as e:
                print(f"Warning: Error reading existing attendance log: {e}")
                # Decide if you want to continue or stop if reading fails

        # Create file with header if it doesn't exist or is empty
        if not file_exists or os.path.getsize(attendance_file_path) == 0:
            try:
                with open(attendance_file_path, 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(attendance_header)
                print(f"Initialized attendance file: {attendance_file_path}")
            except Exception as e:
                 mess.showerror("File Error", f"Could not create/write header to attendance file:\n{e}")
                 return # Stop if we can't write to the log

        message_label.config(text="Starting attendance tracking... Press 'q' in the video window to stop.")
        window.update_idletasks()

        # --- Tracking Loop ---
        active_tracking = True
        last_sent_code = None # Track last sent code to avoid flooding Arduino
        last_code_time = time.time()
        send_interval = 0.5 # Minimum seconds between sending the same code

        while active_tracking:
            ret, im = cam.read()
            if not ret:
                print("Warning: Failed to grab frame. Retrying...")
                time.sleep(0.1) # Avoid busy-waiting
                continue

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))

            current_time_check = time.time()
            detected_code = None # Code for this frame, determined by highest confidence face

            # Process detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2) # Draw rectangle

                # Recognize the face
                serial_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                student_id_str = str(serial_id) # ID from recognizer

                current_time = datetime.datetime.now()
                date_str = current_time.strftime("%d-%m-%Y")
                time_str = current_time.strftime("%H:%M:%S")

                display_text = "Unknown"
                face_code = b'A' # Default to Absent/Unknown for this face

                if conf < CONFIDENCE_THRESHOLD: # Confidence check (lower value means better match)
                    try:
                        # Find student info in the dataframe using the string ID
                        student_info = df.loc[df['ID'] == student_id_str]

                        if not student_info.empty:
                            student_name = student_info['Name'].iloc[0] # Use iloc[0] for first match
                            # Calculate confidence percentage (100 = perfect match)
                            confidence_percent = round(100 - conf)
                            display_text = f"{student_name} ({confidence_percent}%)"
                            face_code = b'P' # Recognized -> Present

                            # Record attendance only ONCE per session for this ID
                            if student_id_str not in logged_ids_today:
                                attendance_data = [student_id_str, student_name, date_str, time_str]
                                try:
                                    with open(attendance_file_path, 'a', newline='') as csvFile: # Use 'a' (append)
                                        writer = csv.writer(csvFile)
                                        writer.writerow(attendance_data)
                                    logged_ids_today.add(student_id_str) # Mark as logged for this session
                                    print(f"Attendance Recorded: ID={student_id_str}, Name={student_name}")
                                    message_label.config(text=f"Welcome {student_name}! Attendance recorded.")
                                    # Optional: Refresh TreeView immediately (can slow down if many entries)
                                    # tv.insert('', 'end', values=(student_id_str, student_name, date_str, time_str))
                                    # tv.yview_moveto(1) # Scroll to bottom

                                except Exception as e:
                                     print(f"Error writing attendance record: {e}")
                                     message_label.config(text="Error saving attendance record.")
                            # else: # Already logged today
                            #     display_text += " (Logged)" # Optional: indicate already logged
                        else:
                             # ID recognized by model but not found in CSV details file
                             display_text = f"ID {student_id_str} (Not Registered)"
                             face_code = b'A'

                    except Exception as e:
                         print(f"Error processing recognized face ID {student_id_str}: {e}")
                         display_text = "Processing Error"
                         face_code = b'A'
                else:
                    # Confidence is too low (>= threshold) - treat as Unknown
                    display_text = "Unknown"
                    face_code = b'A'

                # --- Determine the overall code to send for this frame ---
                # Priority: If any 'P' detected, send 'P'. Otherwise, send 'A'.
                if face_code == b'P':
                    detected_code = b'P'
                elif detected_code is None: # If no 'P' found yet, set to 'A'
                    detected_code = b'A'

                # Put text on image (above rectangle)
                cv2.putText(im, str(display_text), (x + 5, y - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Send Code to Arduino (After processing all faces in the frame) ---
            final_code_to_send = detected_code if detected_code else b'I' # Send Idle if no face detected
            # Send only if the code changed OR enough time passed since last send
            if final_code_to_send != last_sent_code or (current_time_check - last_code_time > send_interval):
                 send_to_arduino(final_code_to_send)
                 last_sent_code = final_code_to_send
                 last_code_time = current_time_check


            # Display the resulting frame
            cv2.imshow('Attendance Tracking - Press "q" to Stop', im)

            # --- Check for Quit Key ---
            key = cv2.waitKey(10) & 0xFF # Check keys every 10ms
            if key == ord('q'):
                active_tracking = False
                print("Stopping attendance tracking...")
                message_label.config(text="Attendance tracking stopped by user.")
                send_to_arduino(b'I') # Send Idle state when stopping
                break # Exit the while loop

    except IOError as e:
         print(f"Camera Error: {e}")
         mess.showerror("Camera Error", f"Failed to access the camera (Index {CAMERA_INDEX}).\nIs it connected and not in use?\nError: {e}")
         message_label.config(text="Camera error occurred during tracking.")
    except pd.errors.EmptyDataError: # Should be caught earlier, but good backup
         mess.showerror("CSV Error", f"StudentDetails.csv is empty or invalid.")
         message_label.config(text="Student details file empty.")
    except cv2.error as cv_err:
         print(f"OpenCV error during tracking: {cv_err}")
         mess.showerror("OpenCV Error", f"An OpenCV error occurred during tracking:\n{cv_err}")
         message_label.config(text="Tracking failed (OpenCV error).")
    except Exception as e:
         print(f"An unexpected error occurred during tracking: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback for debugging
         mess.showerror("Tracking Error", f"An unexpected error occurred during tracking:\n{e}")
         message_label.config(text="Tracking failed unexpectedly.")
    finally:
        # --- Cleanup ---
        if cam and cam.isOpened():
            cam.release()
        cv2.destroyAllWindows()
        print("Tracking resources released.")
        # Send Idle state again just in case it was missed
        send_to_arduino(b'I')
        # Refresh log display after tracking session ends
        display_attendance_log()


# --- Arduino Data Processing ---
# (Optional: Implement if Arduino sends data BACK to Python)
def process_arduino_data():
    """Reads and processes data received FROM Arduino."""
    global arduino, arduino_status
    if arduino and arduino.in_waiting > 0:
        try:
            data = arduino.readline().decode('utf-8', errors='ignore').strip()
            if data: # Process only if data is not empty
                print(f"Received from Arduino: {data}") # Debug print

                # --- Add logic here to handle specific messages from Arduino ---
                if "Status:SensorFailure" in data:
                    mess.showwarning("Arduino Alert", "Arduino reported a potential sensor failure!")
                    arduino_status = f"Arduino: Sensor Alert! ({ARDUINO_PORT})"
                    arduino_status_label.config(text=arduino_status)
                elif "Status:OK" in data:
                    arduino_status = f"Arduino: OK ({ARDUINO_PORT})"
                    arduino_status_label.config(text=arduino_status)
                # Example: elif data.startswith("LDR:"): value = data.split(':')[1]; update_ldr_label(value)
                # ------------------------------------------------------------

        except UnicodeDecodeError:
            # Should be handled by errors='ignore', but good practice
            print("Warning: Received non-UTF-8 data from Arduino.")
        except serial.SerialException as e:
             print(f"Arduino communication error while reading: {e}")
             mess.showerror("Arduino Error", f"Arduino communication lost: {e}")
             arduino_status = "Arduino: Disconnected"
             arduino_status_label.config(text=arduino_status)
             try:
                 if arduino: arduino.close()
             except: pass
             arduino = None # Set to None to stop further read attempts
        except Exception as e:
            print(f"Error processing Arduino data: {e}")

def check_arduino_data():
    """Periodically checks for incoming Arduino data."""
    if arduino: # Only process if connection is supposed to be active
        process_arduino_data()
    # Schedule the next check - Adjust interval (ms) as needed
    window.after(200, check_arduino_data) # Check again in 200ms


def on_closing():
    """Handles window closing event."""
    print("Closing application...")
    if mess.askokcancel("Quit", "Do you want to exit the application?"):
        # Send final Idle signal to Arduino
        send_to_arduino(b'I')
        time.sleep(0.1) # Brief pause to allow sending

        # Close Arduino connection if open
        if arduino and arduino.is_open:
            try:
                arduino.close()
                print("Arduino connection closed.")
            except Exception as e:
                print(f"Error closing Arduino connection: {e}")

        # Close OpenCV windows just in case
        cv2.destroyAllWindows()

        # Destroy the Tkinter window
        window.destroy()
        print("Application closed.")
    # else: User cancelled closing


# --- GUI Setup (Mostly unchanged, added Arduino Status Label) ---
window = tk.Tk()
window.title("Face Recognition Attendance System")
# Adjust size or use maximize
# window.state('zoomed') # Maximize on Windows
window.geometry("1280x720")
window.configure(background='#2c3e50')

# --- Fonts ---
title_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
label_font = tkFont.Font(family='Helvetica', size=12)
button_font = tkFont.Font(family='Helvetica', size=10, weight='bold')
message_font = tkFont.Font(family='Helvetica', size=10, slant='italic')

# --- Title Frame ---
title_frame = tk.Frame(window, bg='#1abc9c')
title_frame.pack(fill='x')
title_label = tk.Label(title_frame, text="Face Recognition Based Attendance System", font=title_font, fg='white', bg='#1abc9c', pady=10)
title_label.pack()
datetime_frame = tk.Frame(title_frame, bg='#1abc9c')
datetime_frame.pack()
date_label = tk.Label(datetime_frame, text="--DATE--", font=label_font, fg='white', bg='#1abc9c')
date_label.pack(side=tk.LEFT, padx=10)
time_label = tk.Label(datetime_frame, text="--TIME--", font=label_font, fg='white', bg='#1abc9c')
time_label.pack(side=tk.LEFT, padx=10)

# --- Main Content Frame ---
content_frame = tk.Frame(window, bg='#2c3e50')
content_frame.pack(fill='both', expand=True, padx=10, pady=10)

# --- Left Frame (Attendance Log) ---
left_frame = tk.LabelFrame(content_frame, text="Attendance Log", font=label_font, bd=2, relief='groove', bg='#ecf0f1', fg='#2c3e50')
left_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)
tv_frame = tk.Frame(left_frame)
tv_frame.pack(fill='both', expand=True, padx=5, pady=5)
scroll_y = ttk.Scrollbar(tv_frame, orient='vertical')
tv = ttk.Treeview(tv_frame, columns=('ID', 'NAME', 'DATE', 'TIME'), show='headings', yscrollcommand=scroll_y.set, height=15)
scroll_y.pack(side='right', fill='y')
scroll_y.config(command=tv.yview)
tv.heading('ID', text='ID'); tv.heading('NAME', text='NAME'); tv.heading('DATE', text='DATE'); tv.heading('TIME', text='TIME')
tv.column('ID', width=80, anchor='center'); tv.column('NAME', width=150, anchor='w'); tv.column('DATE', width=120, anchor='center'); tv.column('TIME', width=120, anchor='center')
tv.pack(fill='both', expand=True)
button_frame_left = tk.Frame(left_frame, bg='#ecf0f1')
button_frame_left.pack(fill='x', pady=10, padx=5)
btn_track = tk.Button(button_frame_left, text="Take Attendance", command=track_images, font=button_font, bg='#3498db', fg='white', width=15, height=2)
btn_track.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
btn_refresh_log = tk.Button(button_frame_left, text="Refresh Log", command=display_attendance_log, font=button_font, bg='#95a5a6', fg='white', width=15, height=2)
btn_refresh_log.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
btn_clear_log_display = tk.Button(button_frame_left, text="Clear Log Display", command=clear_treeview, font=button_font, bg='#7f8c8d', fg='white', width=15, height=2)
btn_clear_log_display.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
button_frame_left.grid_columnconfigure((0,1), weight=1) # Make buttons expand

# --- Right Frame (New Registrations) ---
right_frame = tk.LabelFrame(content_frame, text="New Registrations", font=label_font, bd=2, relief='groove', bg='#ecf0f1', fg='#2c3e50')
right_frame.pack(side=tk.RIGHT, fill='y', padx=10, pady=10)
lbl_id = tk.Label(right_frame, text="Student ID:", font=label_font, bg='#ecf0f1')
lbl_id.grid(row=0, column=0, padx=10, pady=10, sticky='w')
txt_id = tk.Entry(right_frame, font=label_font, width=15)
txt_id.grid(row=0, column=1, padx=10, pady=10)
# btn_clear_id = tk.Button(right_frame, text="X", command=lambda: txt_id.delete(0, 'end'), font=button_font, width=2)
# btn_clear_id.grid(row=0, column=2, padx=5, pady=10) # Small clear button
lbl_name = tk.Label(right_frame, text="Student Name:", font=label_font, bg='#ecf0f1')
lbl_name.grid(row=1, column=0, padx=10, pady=10, sticky='w')
txt_name = tk.Entry(right_frame, font=label_font, width=15)
txt_name.grid(row=1, column=1, padx=10, pady=10)
# btn_clear_name = tk.Button(right_frame, text="X", command=lambda: txt_name.delete(0, 'end'), font=button_font, width=2)
# btn_clear_name.grid(row=1, column=2, padx=5, pady=10)
btn_clear_all = tk.Button(right_frame, text="Clear Fields", command=clear_id_name, font=button_font, bg='#f39c12', fg='white', width=15)
btn_clear_all.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
separator = ttk.Separator(right_frame, orient='horizontal')
separator.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=15)
lbl_steps = tk.Label(right_frame, text="Step 1: Enter Details & Take Images", font=label_font, bg='#ecf0f1')
lbl_steps.grid(row=4, column=0, columnspan=2, padx=10, pady=5)
btn_take_images = tk.Button(right_frame, text="Take Images", command=take_images, font=button_font, bg='#2ecc71', fg='white', width=20, height=2)
btn_take_images.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
lbl_steps2 = tk.Label(right_frame, text="Step 2: Train Recognizer", font=label_font, bg='#ecf0f1')
lbl_steps2.grid(row=6, column=0, columnspan=2, padx=10, pady=5)
btn_train_images = tk.Button(right_frame, text="Train Images", command=train_images, font=button_font, bg='#e67e22', fg='white', width=20, height=2)
btn_train_images.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
right_frame.grid_columnconfigure((0,1), weight=1) # Make entry/button expand a bit


# --- Bottom Frame (Status Message & Arduino Status) ---
bottom_frame = tk.Frame(window, bg='#34495e', height=30)
bottom_frame.pack(side=tk.BOTTOM, fill='x')
message_label = tk.Label(bottom_frame, text="Welcome! System Initializing...", font=message_font, fg='white', bg='#34495e')
message_label.pack(side=tk.LEFT, padx=10, pady=5)
# <<< Arduino Status Label >>>
arduino_status_label = tk.Label(bottom_frame, text=arduino_status, font=message_font, fg='white', bg='#34495e')
arduino_status_label.pack(side=tk.RIGHT, padx=10, pady=5)


# --- Initial Setup Calls ---
if check_haarcascadefile(): # Only proceed if cascade exists
    display_attendance_log() # Load today's log on startup
    update_clock() # Start the clock update loop
    # Start polling Arduino for incoming data IF connected
    if arduino:
        check_arduino_data()
    message_label.config(text="System Ready.") # Update status after init
else:
     message_label.config(text="Critical Error: Haar Cascade missing. Application halted.")

# Bind window closing event
window.protocol("WM_DELETE_WINDOW", on_closing)

# --- Start GUI ---
print("Starting Tkinter main loop...")
window.mainloop()

# --- Post-GUI Actions (usually after window.destroy()) ---
print("Application has finished.")