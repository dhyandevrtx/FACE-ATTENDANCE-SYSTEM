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

# --- Email Libraries ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# --- Configuration ---
# !!! IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S PORT !!!
ARDUINO_PORT = '/dev/ttyUSB0' # Or 'COM3', '/dev/cu.usbmodemXXXX', etc.
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

# --- Email Configuration ---
# !!! WARNING: Storing credentials here is insecure for production. Use App Passwords for Gmail. !!!
# !!! Consider using environment variables or a config file instead.                   !!!
EMAIL_SENDER = 'your_email@gmail.com'  # <<< CHANGE THIS - Your email address
EMAIL_PASSWORD = 'your_app_password'   # <<< CHANGE THIS - Your App Password (for Gmail) or regular password (less secure)
EMAIL_RECIPIENT = 'recipient_email@example.com' # <<< CHANGE THIS - Where to send the report
SMTP_SERVER = 'smtp.gmail.com'       # Common for Gmail
SMTP_PORT = 587                      # Common for Gmail (TLS)

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
    # Send an initial 'Idle' state code ('I') to Arduino
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
    mess.showerror("Arduino Error", f"Could not connect to Arduino on {ARDUINO_PORT}.\n{e}\n\nPlease check:\n1. Is it plugged in?\n2. Is the correct port selected?\n3. Is the port free (Serial Monitor closed)?\n\nHardware feedback will be disabled.")
    arduino = None # Ensure it's None if connection failed
    arduino_status = "Arduino: Connection Failed"
except Exception as e:
    print(f"--- An unexpected error occurred during Arduino connection ---")
    print(f"{e}")
    mess.showerror("Arduino Error", f"An unexpected error occurred during Arduino connection.\n{e}\n\nHardware feedback will be disabled.")
    arduino = None
    arduino_status = "Arduino: Error"


# --- Function Definitions ---

def check_haarcascadefile():
    """Checks if the Haar Cascade file exists."""
    exists = os.path.isfile(haar_cascade_file)
    if exists: pass
    else: mess.showerror('Error', f'Critical Error: Haar Cascade file not found!\n{haar_cascade_file}'); window.destroy()
    return exists

def is_number(s):
    """Checks if a string represents a number."""
    try: float(s); return True
    except ValueError: return False

def update_clock():
    """Updates the date and time labels periodically."""
    now = time.strftime("%d-%b-%Y %H:%M:%S")
    date_label.config(text=now.split()[0])
    time_label.config(text=now.split()[1])
    window.after(1000, update_clock)

def clear_id_name():
    """Clears the ID and Name entry fields."""
    txt_id.delete(0, 'end'); txt_name.delete(0, 'end')
    message_label.config(text="Registration fields cleared.")

def clear_treeview():
    """Clears the attendance log display (Treeview)."""
    count = 0
    for item in tv.get_children(): tv.delete(item); count += 1
    message_label.config(text=f"Log display cleared ({count} items removed).")

def display_attendance_log():
    """Loads and displays today's attendance log in the Treeview."""
    clear_treeview()
    today_date = datetime.date.today().strftime("%d-%m-%Y")
    attendance_file_path = os.path.join(attendance_folder, f"Attendance_{today_date}.csv")

    if not os.path.isfile(attendance_file_path):
        message_label.config(text=f"No attendance log for today ({today_date}).")
        return

    try:
        count = 0
        # Use latin-1 and ignore errors for robust reading
        with open(attendance_file_path, 'r', newline='', encoding='latin-1', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None: message_label.config(text=f"Attendance file for {today_date} empty."); return
            for row in reader:
                if len(row) >= 4: tv.insert('', 'end', values=(row[0], row[1], row[2], row[3])); count += 1
            message_label.config(text=f"Displayed {count} records for {today_date}.")
    except Exception as e:
        mess.showerror("CSV Error", f"Error reading attendance file:\n{attendance_file_path}\n\nError: {e}")
        message_label.config(text="Error loading attendance log.")

def send_to_arduino(code_byte):
    """Sends a single byte code to Arduino if connected."""
    global arduino, arduino_status
    if arduino:
        try: arduino.write(code_byte)
        except serial.SerialTimeoutException: print(f"Warning: Timeout writing to Arduino.")
        except serial.SerialException as e:
            print(f"--- Arduino Send Error: {e} ---")
            mess.showerror("Arduino Error", f"Lost connection sending data: {e}")
            arduino_status = "Arduino: Disconnected"; arduino_status_label.config(text=arduino_status)
            try: arduino.close()
            except: pass
            arduino = None
        except Exception as e: print(f"--- Unexpected Arduino Send Error: {e} ---")

def take_images():
    """Captures images for a new student registration."""
    if not check_haarcascadefile(): return
    student_id = txt_id.get().strip(); student_name = txt_name.get().strip()

    if not student_id or not student_name: mess.showerror('Input Error', 'ID and Name required.'); return
    if not is_number(student_id): mess.showerror('Input Error', 'ID must be numeric.'); return
    if not all(c.isalpha() or c.isspace() for c in student_name): mess.showerror('Input Error', 'Name: letters/spaces only.'); return

    cam = None
    try:
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened(): raise IOError(f"Cannot open webcam {CAMERA_INDEX}")
        face_detector = cv2.CascadeClassifier(haar_cascade_file)
        student_id_int = int(student_id)

        existing_ids = set()
        if os.path.isfile(student_details_file):
            try: # Try reading details file robustly
                try: df_details = pd.read_csv(student_details_file, encoding='utf-8')
                except UnicodeDecodeError: df_details = pd.read_csv(student_details_file, encoding='latin-1', errors='ignore')
                if not df_details.empty: existing_ids.update(df_details['ID'].astype(int).tolist())
            except Exception as e: mess.showwarning("CSV Warning", f"Could not read student details: {e}")

        if student_id_int in existing_ids:
             if not mess.askyesno("ID Exists", f"ID {student_id} exists. Overwrite images?"): message_label.config(text="Capture cancelled."); return
             else: print(f"Overwriting images for ID: {student_id}")
        else: # Add new details
             row = [student_id, student_name]
             try: # Write details file using UTF-8
                  file_exists = os.path.isfile(student_details_file)
                  with open(student_details_file, 'a+', newline='', encoding='utf-8') as csvFile:
                       writer = csv.writer(csvFile)
                       if not file_exists or os.path.getsize(student_details_file) == 0: writer.writerow(['ID', 'Name'])
                       writer.writerow(row)
                  print(f"New details saved: ID={student_id}, Name={student_name}")
             except Exception as e: mess.showerror("Save Error", f"Could not save details: {e}"); # Continue capture? return?

        message_label.config(text=f"Taking images for {student_name}..."); window.update_idletasks()
        img_count = 0; max_images = 100
        while img_count < max_images:
            ret, img = cam.read()
            if not ret: print("Warning: Failed frame grab."); time.sleep(0.1); continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2); img_count += 1
                face_img = gray[y:y + h, x:x + w]
                img_path = os.path.join(training_image_folder, f"{student_name}.{student_id}.{img_count}.jpg")
                if not os.path.exists(training_image_folder): os.makedirs(training_image_folder)
                cv2.imwrite(img_path, face_img)
                cv2.putText(img, f"Capture: {img_count}/{max_images}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow('Taking Images - Press "q" to Cancel', img)
            if cv2.waitKey(50) & 0xFF == ord('q'): message_label.config(text="Capture cancelled."); print("Capture cancelled."); img_count = max_images + 1; break

        if img_count == max_images: message_label.config(text=f"Captured {max_images} images."); print(f"Captured {max_images} images."); mess.showinfo("Capture Complete", f"{max_images} images for ID: {student_id}")
        elif img_count < max_images: message_label.config(text=f"Capture stopped ({img_count} saved)."); print(f"Warning: Capture stopped ({img_count} saved).")
    except Exception as e: mess.showerror("Error", f"Capture error:\n{e}"); message_label.config(text="Capture failed.")
    finally:
        if cam and cam.isOpened(): cam.release()
        cv2.destroyAllWindows(); print("Capture resources released.")

def train_images():
    """Trains the LBPH face recognizer."""
    if not check_haarcascadefile(): return
    image_paths = [os.path.join(training_image_folder, f) for f in os.listdir(training_image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_paths: mess.showerror("Training Error", "No images found for training."); message_label.config(text="Training failed: No images."); return

    faces = []; ids = []; message_label.config(text="Starting training..."); window.update_idletasks()
    total_images = 0; skipped_files = 0; processed_ids = set()
    for imagePath in image_paths:
        try:
            pil_image = Image.open(imagePath).convert('L'); image_np = np.array(pil_image, dtype='uint8')
            filename = os.path.split(imagePath)[-1]; parts = filename.split(".")
            if len(parts) < 3: print(f"Skip bad format: {filename}"); skipped_files += 1; continue
            img_id = int(parts[1])
            faces.append(image_np); ids.append(img_id); processed_ids.add(img_id); total_images += 1
        except Exception as e: print(f"Skip {imagePath}: {e}"); skipped_files += 1
    if not faces or not ids: mess.showerror("Training Error", "No valid images processed."); message_label.config(text="Training failed: No valid images."); return

    ids = np.array(ids)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("Training LBPH..."); recognizer.train(faces, ids)
        if not os.path.exists(training_label_folder): os.makedirs(training_label_folder)
        recognizer.write(trainer_file)
        num_students = len(processed_ids)
        print(f"Trained {total_images}/{len(image_paths)} images for {num_students} students. Skipped: {skipped_files}")
        mess.showinfo("Training Complete", f"Trained {total_images} images for {num_students} students.\nSkipped: {skipped_files}\nModel: {trainer_file}")
        message_label.config(text="Training complete.")
    except Exception as e: print(f"Train/Save Error: {e}"); mess.showerror("Training Error", f"Error during training/saving:\n{e}"); message_label.config(text="Training failed.")

def track_images():
    """Tracks faces, recognizes, logs attendance, sends Arduino signals."""
    global arduino
    if not check_haarcascadefile(): return
    if not os.path.isfile(trainer_file): mess.showerror("Tracking Error", f"Trainer missing:\n{trainer_file}"); return
    if not os.path.isfile(student_details_file): mess.showerror("Tracking Error", f"Details missing:\n{student_details_file}"); return

    cam = None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(); recognizer.read(trainer_file)
        face_cascade = cv2.CascadeClassifier(haar_cascade_file)
        try: # Read details file robustly
            try: df = pd.read_csv(student_details_file, encoding='utf-8')
            except UnicodeDecodeError: df = pd.read_csv(student_details_file, encoding='latin-1')
            if df.empty: raise pd.errors.EmptyDataError
            df['ID'] = df['ID'].astype(str)
        except Exception as e: mess.showerror("CSV Error", f"Error reading details CSV:\n{e}"); return
        print("Recognizer, cascade, details loaded.")

        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened(): raise IOError(f"Cannot open webcam {CAMERA_INDEX}")
        cam.set(3, 640); cam.set(4, 480); font = cv2.FONT_HERSHEY_SIMPLEX

        today_date = datetime.date.today().strftime("%d-%m-%Y")
        attendance_file_path = os.path.join(attendance_folder, f"Attendance_{today_date}.csv")
        attendance_header = ['ID', 'Name', 'Date', 'Time']; file_exists = os.path.isfile(attendance_file_path)
        logged_ids_today = set()
        if file_exists and os.path.getsize(attendance_file_path) > 0:
            try: # Read existing attendance robustly
                attendance_df = pd.read_csv(attendance_file_path, encoding='latin-1', errors='ignore')
                if not attendance_df.empty and 'ID' in attendance_df.columns: logged_ids_today.update(attendance_df['ID'].astype(str).tolist())
            except Exception as e: print(f"Warn: Error reading existing attendance: {e}")
        if not file_exists or os.path.getsize(attendance_file_path) == 0:
            try: # Write header using UTF-8
                with open(attendance_file_path, 'w', newline='', encoding='utf-8') as csvFile: csv.writer(csvFile).writerow(attendance_header)
                print(f"Initialized attendance file: {attendance_file_path}")
            except Exception as e: mess.showerror("File Error", f"Could not init attendance file:\n{e}"); return

        message_label.config(text="Tracking... Press 'q' to Stop."); window.update_idletasks()
        active_tracking = True; last_sent_code = None; last_code_time = time.time(); send_interval = 0.5

        while active_tracking:
            ret, im = cam.read()
            if not ret: print("Warn: Failed frame grab."); time.sleep(0.1); continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30,30))
            current_time_check = time.time(); detected_code = None

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial_id, conf = recognizer.predict(gray[y:y + h, x:x + w]); student_id_str = str(serial_id)
                ts = time.time(); date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y'); time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                display_text = "Unknown"; face_code = b'A'

                if conf < CONFIDENCE_THRESHOLD:
                    try:
                        student_info = df.loc[df['ID'] == student_id_str]
                        if not student_info.empty:
                            student_name = student_info['Name'].iloc[0]; conf_percent = round(100 - conf)
                            display_text = f"{student_name} ({conf_percent}%)"; face_code = b'P'
                            if student_id_str not in logged_ids_today:
                                attendance_data = [student_id_str, student_name, date_str, time_str]
                                try: # Write attendance record using UTF-8
                                    with open(attendance_file_path, 'a', newline='', encoding='utf-8') as csvFile: csv.writer(csvFile).writerow(attendance_data)
                                    logged_ids_today.add(student_id_str); print(f"Recorded: ID={student_id_str}"); message_label.config(text=f"Welcome {student_name}!")
                                except Exception as e: print(f"Error writing record: {e}"); message_label.config(text="Error saving record.")
                        else: display_text = f"ID {student_id_str} (Not Registered)"; face_code = b'A'
                    except Exception as e: print(f"Error processing ID {student_id_str}: {e}"); display_text = "Error"; face_code = b'A'
                else: display_text = "Unknown"; face_code = b'A'

                if face_code == b'P': detected_code = b'P'
                elif detected_code is None: detected_code = b'A'
                cv2.putText(im, display_text, (x + 5, y - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            final_code_to_send = detected_code if detected_code else b'I'
            if final_code_to_send != last_sent_code or (current_time_check - last_code_time > send_interval):
                 send_to_arduino(final_code_to_send); last_sent_code = final_code_to_send; last_code_time = current_time_check

            cv2.imshow('Attendance Tracking - Press "q" to Stop', im)
            if cv2.waitKey(10) & 0xFF == ord('q'): active_tracking = False; print("Stopping tracking..."); message_label.config(text="Tracking stopped."); send_to_arduino(b'I'); break
    except Exception as e: import traceback; traceback.print_exc(); mess.showerror("Tracking Error", f"Unexpected tracking error:\n{e}"); message_label.config(text="Tracking failed.")
    finally:
        if cam and cam.isOpened(): cam.release()
        cv2.destroyAllWindows(); print("Tracking resources released."); send_to_arduino(b'I')
        display_attendance_log()

def generate_and_email_report():
    """Generates and emails the attendance report for today."""
    print("Attempting email report..."); message_label.config(text="Generating email..."); window.update_idletasks()
    today_date = datetime.date.today().strftime("%d-%m-%Y")
    attendance_file_path = os.path.join(attendance_folder, f"Attendance_{today_date}.csv")

    if not os.path.isfile(attendance_file_path): mess.showerror("Email Error", f"File not found:\n{attendance_file_path}"); message_label.config(text="Email fail: File missing."); return
    if os.path.getsize(attendance_file_path) <= 50: mess.showinfo("Email Info", f"Attendance file empty for {today_date}."); message_label.config(text="Email cancelled: File empty."); return

    try: # Construct and send email
        msg = MIMEMultipart(); msg['From'] = EMAIL_SENDER; msg['To'] = EMAIL_RECIPIENT; msg['Subject'] = f"Attendance Report - {today_date}"
        body = f"Attendance report for {today_date}.\n\nGenerated by Face Recognition System."
        msg.attach(MIMEText(body, 'plain'))

        filename = os.path.basename(attendance_file_path)
        try:
            with open(attendance_file_path, "rb") as attachment: part = MIMEBase('application', 'octet-stream'); part.set_payload(attachment.read())
            encoders.encode_base64(part); part.add_header('Content-Disposition', f"attachment; filename= {filename}"); msg.attach(part)
            print(f"Attached: {filename}")
        except Exception as attach_err: mess.showerror("Email Error", f"Attach failed: {attach_err}"); message_label.config(text="Email fail: Attach error."); return

        print(f"Connecting to SMTP: {SMTP_SERVER}:{SMTP_PORT}"); server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls(); print("Logging in..."); server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string(); print(f"Sending to {EMAIL_RECIPIENT}..."); server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, text)
        server.quit(); print("Email sent."); mess.showinfo("Email Sent", f"Report sent to {EMAIL_RECIPIENT}."); message_label.config(text="Email sent.")
    except smtplib.SMTPAuthenticationError: mess.showerror("Email Error", "Authentication Failed. Check email/password (or App Password)."); message_label.config(text="Email fail: Auth error.")
    except Exception as e: import traceback; traceback.print_exc(); mess.showerror("Email Error", f"Email failed:\n{e}"); message_label.config(text="Email failed.")

def process_arduino_data():
    """Reads and processes data received FROM Arduino."""
    global arduino, arduino_status
    if arduino and arduino.in_waiting > 0:
        try:
            data = arduino.readline().decode('utf-8', errors='ignore').strip()
            if data:
                print(f"Arduino >> {data}") # Debug print received data
                if "Status:SensorFailure" in data:
                    mess.showwarning("Arduino Alert", "Sensor Failure Reported!"); arduino_status = f"Arduino: Sensor Alert! ({ARDUINO_PORT})"
                elif "Status:OK" in data: arduino_status = f"Arduino: OK ({ARDUINO_PORT})"
                # Add more conditions here to parse LDR: value if needed for GUI display
                # elif data.startswith("LDR:"): try: ldr_val = int(data.split(':')[1]); print(f"LDR Value: {ldr_val}") except: pass
                arduino_status_label.config(text=arduino_status) # Update status label
        except Exception as e: print(f"Error processing Arduino data: {e}")

def check_arduino_data():
    """Periodically checks for incoming Arduino data."""
    if arduino: process_arduino_data()
    window.after(200, check_arduino_data) # Check again in 200ms

def on_closing():
    """Handles window closing event."""
    if mess.askokcancel("Quit", "Do you want to exit?"):
        print("Closing application..."); send_to_arduino(b'I'); time.sleep(0.1)
        if arduino and arduino.is_open:
            try: arduino.close(); print("Arduino connection closed.")
            except Exception as e: print(f"Error closing Arduino: {e}")
        cv2.destroyAllWindows(); window.destroy(); print("Application closed.")

# --- GUI Setup ---
window = tk.Tk(); window.title("Face Recognition Attendance System"); window.geometry("1280x720"); window.configure(background='#2c3e50')
title_font=tkFont.Font(family='Helvetica', size=20, weight='bold'); label_font=tkFont.Font(family='Helvetica', size=12)
button_font=tkFont.Font(family='Helvetica', size=10, weight='bold'); message_font=tkFont.Font(family='Helvetica', size=10, slant='italic')

title_frame = tk.Frame(window, bg='#1abc9c'); title_frame.pack(fill='x')
tk.Label(title_frame, text="Face Recognition Based Attendance System", font=title_font, fg='white', bg='#1abc9c', pady=10).pack()
datetime_frame = tk.Frame(title_frame, bg='#1abc9c'); datetime_frame.pack()
date_label = tk.Label(datetime_frame, text="--DATE--", font=label_font, fg='white', bg='#1abc9c'); date_label.pack(side=tk.LEFT, padx=10)
time_label = tk.Label(datetime_frame, text="--TIME--", font=label_font, fg='white', bg='#1abc9c'); time_label.pack(side=tk.LEFT, padx=10)

content_frame = tk.Frame(window, bg='#2c3e50'); content_frame.pack(fill='both', expand=True, padx=10, pady=10)

left_frame = tk.LabelFrame(content_frame, text="Attendance Log", font=label_font, bd=2, relief='groove', bg='#ecf0f1', fg='#2c3e50')
left_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)
tv_frame = tk.Frame(left_frame); tv_frame.pack(fill='both', expand=True, padx=5, pady=5)
scroll_y = ttk.Scrollbar(tv_frame, orient='vertical')
tv = ttk.Treeview(tv_frame, columns=('ID', 'NAME', 'DATE', 'TIME'), show='headings', yscrollcommand=scroll_y.set, height=15)
scroll_y.pack(side='right', fill='y'); scroll_y.config(command=tv.yview)
tv.heading('ID', text='ID'); tv.column('ID', width=80, anchor='center')
tv.heading('NAME', text='NAME'); tv.column('NAME', width=150, anchor='w')
tv.heading('DATE', text='DATE'); tv.column('DATE', width=120, anchor='center')
tv.heading('TIME', text='TIME'); tv.column('TIME', width=120, anchor='center')
tv.pack(fill='both', expand=True)

button_frame_left = tk.Frame(left_frame, bg='#ecf0f1'); button_frame_left.pack(fill='x', pady=5, padx=5)
tk.Button(button_frame_left, text="Take Attendance", command=track_images, font=button_font, bg='#3498db', fg='white', height=2).grid(row=0, column=0, padx=5, pady=2, sticky='ew')
tk.Button(button_frame_left, text="Refresh Log", command=display_attendance_log, font=button_font, bg='#95a5a6', fg='white', height=2).grid(row=0, column=1, padx=5, pady=2, sticky='ew')
tk.Button(button_frame_left, text="Clear Log Display", command=clear_treeview, font=button_font, bg='#7f8c8d', fg='white', height=2).grid(row=1, column=0, padx=5, pady=2, sticky='ew')
tk.Button(button_frame_left, text="Email Today's Report", command=generate_and_email_report, font=button_font, bg='#16a085', fg='white', height=2).grid(row=1, column=1, padx=5, pady=2, sticky='ew')
button_frame_left.grid_columnconfigure((0,1), weight=1)

right_frame = tk.LabelFrame(content_frame, text="New Registrations", font=label_font, bd=2, relief='groove', bg='#ecf0f1', fg='#2c3e50')
right_frame.pack(side=tk.RIGHT, fill='y', padx=10, pady=10)
tk.Label(right_frame, text="Student ID:", font=label_font, bg='#ecf0f1').grid(row=0, column=0, padx=10, pady=10, sticky='w')
txt_id = tk.Entry(right_frame, font=label_font, width=15); txt_id.grid(row=0, column=1, padx=10, pady=10)
tk.Label(right_frame, text="Student Name:", font=label_font, bg='#ecf0f1').grid(row=1, column=0, padx=10, pady=10, sticky='w')
txt_name = tk.Entry(right_frame, font=label_font, width=15); txt_name.grid(row=1, column=1, padx=10, pady=10)
tk.Button(right_frame, text="Clear Fields", command=clear_id_name, font=button_font, bg='#f39c12', fg='white').grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
ttk.Separator(right_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=15)
tk.Label(right_frame, text="Step 1: Enter Details & Take Images", font=label_font, bg='#ecf0f1').grid(row=4, column=0, columnspan=2, padx=10, pady=5)
tk.Button(right_frame, text="Take Images", command=take_images, font=button_font, bg='#2ecc71', fg='white', height=2).grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
tk.Label(right_frame, text="Step 2: Train Recognizer", font=label_font, bg='#ecf0f1').grid(row=6, column=0, columnspan=2, padx=10, pady=5)
tk.Button(right_frame, text="Train Images", command=train_images, font=button_font, bg='#e67e22', fg='white', height=2).grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
right_frame.grid_columnconfigure((0,1), weight=1)

bottom_frame = tk.Frame(window, bg='#34495e', height=30); bottom_frame.pack(side=tk.BOTTOM, fill='x')
message_label = tk.Label(bottom_frame, text="Initializing...", font=message_font, fg='white', bg='#34495e'); message_label.pack(side=tk.LEFT, padx=10, pady=5)
arduino_status_label = tk.Label(bottom_frame, text=arduino_status, font=message_font, fg='white', bg='#34495e'); arduino_status_label.pack(side=tk.RIGHT, padx=10, pady=5)

# --- Initial Setup Calls & Main Loop ---
if check_haarcascadefile():
    display_attendance_log(); update_clock()
    if arduino: check_arduino_data() # Start polling Arduino if connected
    message_label.config(text="System Ready.")
else: message_label.config(text="Critical Error: Haar Cascade missing.")
window.protocol("WM_DELETE_WINDOW", on_closing)
print("Starting Tkinter main loop..."); window.mainloop(); print("Application finished.")