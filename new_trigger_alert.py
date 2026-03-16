import yaml
from datetime import datetime
import time
import threading
import os
import sys
import subprocess
# import gi  # Commented for Windows testing - only needed on Linux EdgeAI device
# import serial  # Commented for Windows testing - only needed on Linux EdgeAI device with GSM
# from trigger_config import set_gpio_high, set_gpio_low  # GPIO only on Linux

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_AVAILABLE = True
except ImportError:
    print("[WARNING] GStreamer not available - audio alerts disabled (Windows mode)")
    GST_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    print("[WARNING] Serial/GSM not available - SMS/Call alerts disabled (Windows mode)")
    SERIAL_AVAILABLE = False

# setup
sys.path.append(os.path.abspath('..'))

# gi.require_version('Gst', '1.0')  # Moved to import section
# from gi.repository import Gst

# Initialize GStreamer (only if available)
if GST_AVAILABLE:
    Gst.init(None)


# load configuration
with open("new_config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

try:
    flow = config.get('flow', [])
    if len(flow) != 3:
        print("Invalid flow configuration. The flow value should be a list of three elements.")
        exit()

    config = config.get('config',{})
    current_config = {}
    alert_config = {}

    current_config = config.get(flow[0], {})
    alert_config = current_config.get(flow[2], {})

    # Action configurations
    config_action_yawn = alert_config.get("action", {}).get("yawn", {})
    # print(config_action_yawn)
    config_action_eyes_closed = alert_config.get("action", {}).get("eyes_closed", {})
    config_action_distraction = alert_config.get("action", {}).get("distraction", {})
    config_action_phone = alert_config.get("action", {}).get("phone", {})
    config_action_smoking = alert_config.get("action", {}).get("smoking", {})

    # Params configurations
    params_config = alert_config.get('params', {})
    led_duration_yawn = params_config.get("led_duration", {}).get("yawn")
    # print(led_duration_yawn)
    led_duration_eyes_closed = params_config.get("led_duration", {}).get("eyes_closed")
    led_duration_distraction = params_config.get("led_duration", {}).get("distraction")
    led_duration_phone = params_config.get("led_duration", {}).get("phone")
    led_duration_smoking = params_config.get("led_duration", {}).get("smoking")

    audio_file_path_yawn = params_config.get("audio_file_path", {}).get("yawn")
    audio_file_path_eyes_closed = params_config.get("audio_file_path", {}).get("eyes_closed")
    audio_file_path_distraction = params_config.get("audio_file_path", {}).get("distraction")
    audio_file_path_phone = params_config.get("audio_file_path", {}).get("phone")
    audio_file_path_smoking = params_config.get("audio_file_path", {}).get("smoking")
    # print(audio_file_path_smoking)

    overlap_flag_beep = params_config.get("overlap", {}).get("beep")
    # print(overlap_flag_beep)
    overlap_flag_led = params_config.get("overlap", {}).get("led")
    overlap_flag_sms = params_config.get("overlap", {}).get("sms")
    overlap_flag_call = params_config.get("overlap", {}).get("call")

except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Invalid configuration file. Please check the structure and values.")
    exit()


# To check if the same functions is already the in the thread or not
flag_beep = 0
flag_led = 0
flag_sms = 0
flag_call = 0
flag_warning_light = 0  # AIS-184 warning light flag
buzzer_active = False  # AIS-184 continuous buzzer control flag

# Check SPEAKER
def check_speaker():
    """Checks if the speaker is working by verifying audio device presence."""
    # aplay is Linux-only, skip on Windows
    if os.name == 'nt':  # Windows
        print("[TEST MODE] Skipping speaker check on Windows")
        return True
    
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
        if "card" in result.stdout:
            print("[OK] Speaker detected and available.")
            return True
        else:
            print("[WARNING] No audio device found!")
            return False
    except subprocess.CalledProcessError:
        print("[WARNING] Error checking audio device!")
        return False

# Play audio     
def play_audio(audio_path, decibal=3.0):
    """Plays the corresponding .wav file for the detected label using GStreamer (Linux) or winsound (Windows)."""
    if not GST_AVAILABLE:
        # Windows fallback - play beep FIRST for immediate attention, then WAV file
        print(f"[WINDOWS AUDIO] Playing audio for: {audio_path}")
        
        try:
            import winsound
            
            # STEP 1: Play beep tone for immediate attention
            if 'eye' in audio_path.lower() or 'drowsy' in audio_path.lower() or 'high_beep' in audio_path.lower():
                # High severity - two loud beeps
                print("[DEBUG] Playing HIGH severity beep (2x 1000Hz)")
                for i in range(2):
                    winsound.Beep(1000, 500)  # 1000Hz, 500ms
                    time.sleep(0.1)
                print("[AUDIO] High severity beep completed")
                    
            elif 'distract' in audio_path.lower() or 'focus' in audio_path.lower():
                # Medium severity - one medium beep
                print("[DEBUG] Playing MEDIUM severity beep (800Hz)")
                winsound.Beep(800, 600)  # 800Hz, 600ms
                print("[AUDIO] Medium severity beep completed")
                
            elif 'yawn' in audio_path.lower() or 'break' in audio_path.lower():
                # Low severity - one short beep
                print("[DEBUG] Playing LOW severity beep (600Hz)")
                winsound.Beep(600, 400)  # 600Hz, 400ms
                print("[AUDIO] Low severity beep completed")
                
            else:
                # Default beep
                print("[DEBUG] Playing DEFAULT beep (750Hz)")
                winsound.Beep(750, 500)  # 750Hz, 500ms
                print("[AUDIO] Default beep completed")
            
            # STEP 2: Try to play WAV file if it exists locally
            wav_filename = audio_path.split('/')[-1]  # Get just the filename
            
            if os.path.exists(wav_filename):
                # Play the actual WAV file AFTER the beep
                print(f"[DEBUG] Playing WAV file: {wav_filename}")
                time.sleep(0.2)  # Small pause between beep and WAV
                winsound.PlaySound(wav_filename, winsound.SND_FILENAME)
                print(f"[AUDIO] WAV file played successfully")
            else:
                print(f"[INFO] WAV file not found: {wav_filename}")
            
        except RuntimeError as e:
            # winsound.Beep can raise RuntimeError if system beep is disabled
            print(f"[ERROR] System beep might be disabled: {e}")
            print(f"[FALLBACK] ALERT: {audio_path.split('/')[-1].replace('.wav', '').upper()}")
        except Exception as e:
            print(f"[ERROR] Windows audio playback failed: {e}")
            print(f"[FALLBACK] ALERT: {audio_path.split('/')[-1].replace('.wav', '').upper()}")
        return
    
    # Linux/GStreamer audio playback
    audio_file = audio_path
    print(audio_file)
    if not audio_file or not os.path.exists(audio_file):
        print(f"[ERROR] File not found!")
        return

    print(f"Playing: {audio_file}")
    pipeline = Gst.parse_launch(
        f"filesrc location={audio_file} ! decodebin ! audioconvert ! audioresample ! volume volume={decibal} ! autoaudiosink"
    )
    pipeline.set_state(Gst.State.PLAYING)

    time.sleep(3)  # Adjust based on audio length
    pipeline.set_state(Gst.State.NULL)

def detect_gsm_module(port='/dev/ttyUSB2', baud_rate=115200, timeout=1):
    if not SERIAL_AVAILABLE:
        print("[TEST MODE] GSM module not available (Windows mode)")
        return None
    
    try:
        gsm = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(1)
        gsm.write(b'AT\r')
        time.sleep(1)
        response = gsm.read(gsm.inWaiting()).decode('utf-8')

        if "OK" in response:
            print("[OK] GSM Module detected and responding.")
            return gsm
        else:
            print("⚠️ No response from GSM module. Check connections.")
            return None

    except serial.SerialException as e:
        print(f"❌ Error: {e}")
        return None

def get_gps_location(gps_port='/dev/ttyUSB1', baud_rate=9600, timeout=1):
    if not SERIAL_AVAILABLE:
        print("[TEST MODE] GPS not available - using default location")
        return 13.109744, 80.190466
    
    try:
        gps = serial.Serial(gps_port, baud_rate, timeout=timeout)
        time.sleep(1)

        for _ in range(10):
            line = gps.readline().decode('utf-8', errors='ignore').strip()
            if "GPRMC" in line:
                print(f"📡 Raw GPS Data: {line}")
                data = line.split(',')
                if data[2] == 'A':
                    latitude = convert_to_decimal(data[3], data[4], is_latitude=True)
                    longitude = convert_to_decimal(data[5], data[6], is_latitude=False)
                    gps.close()
                    return latitude, longitude

        gps.close()
#        print("❌ No valid GPS location found.")
        return 13.109744, 80.190466
    except Exception as e:
        print(f"❌ Error getting GPS data: {e}")
        return 13.109744, 80.190466

def convert_to_decimal(coord, direction, is_latitude=True):
    if not coord or not direction:
        return None

    if is_latitude:
        degrees = int(coord[:2])
        minutes = float(coord[2:])
    else:
        degrees = int(coord[:3])
        minutes = float(coord[3:])

    decimal = degrees + (minutes / 60)
    if direction in ['S', 'W']:
        decimal = -decimal
    return round(decimal, 6)

def send_sms(gsm, phone_number, message):
    if gsm:
        try:
            gsm.write(b'AT+CMGF=1\r')
            time.sleep(0.5)
            gsm.write(f'AT+CMGS="{phone_number}"\r'.encode())
            time.sleep(0.5)
            gsm.write(f'{message}\x1A'.encode())
            time.sleep(1.5)
            response = gsm.read(gsm.inWaiting()).decode('utf-8')
            print(f"📨 SMS Response: {response}")
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")

def make_call(gsm, phone_number):
    """
    0 = Active
    1 = Held
    2 = Dialing
    3 = Alerting (ringing)
    4 = Incoming
    5 = Waiting
    6 = Call held by response and waiting
    """

    if gsm:
        try:
            # Start the call
            gsm.write(f'ATD{phone_number};\r'.encode())
            time.sleep(2)

            print("📞 Calling...")
            call_active = True
            track_response = None
            while call_active:
                gsm.write(b'AT+CLCC\r')
                time.sleep(1)
                response = gsm.read(gsm.inWaiting()).decode('utf-8')

                if response == track_response: # check if the response is same as previous
                    call_active = False

                if response:
                    print(f"📲 Call Status: {response.strip()}")

                    if "0,0,0,0,0" in response:
                        print("✅ Call answered.")
                        call_active = False
                    elif "0,0,0,0,6" in response:
                        print("❌ Call ended or failed.")
                        call_active = False
                    elif "NO CARRIER" in response or "BUSY" in response:
                        print("❌ Call not answered / Rejected.")
                        call_active = False

                track_response  = response # tracking response

                time.sleep(2)  # Wait a bit before checking again

        except Exception as e:
            print(f"❌ Error making call: {e}")

# beep function
def beep(file_path = None, decibal=3.0):    
    global flag_beep

    if file_path is None:
        print("Missing file path or duration for alert. Check the Configuration")
        return 0
    print("Entered beep")
    # actual function
    check_speaker()
    play_audio(file_path, decibal)
        
    # test function
    # print("Beep playing - ",file_path)
    # time.sleep(3)
    # print("beep ended")

    flag_beep = 0


# LED function
def led(duration = None):
    global flag_led

    if duration is None:
        print("Missing duration for led. Check the Configuration")
        return 0
    
    # actual function (GPIO only on Linux)
    # set_gpio_high()
    # time.sleep(duration)
    # set_gpio_low()
    print(f"[TEST MODE] Would turn LED on for {duration}s")

    # test function
    # print("Led turned on")
    # time.sleep(duration)
    # print("Led turned off")

    flag_led = 0

# Send SMS
def initiate_sms(PHONE_NUMBER = None, MESSAGE = None):
    global flag_sms

    if PHONE_NUMBER is None or MESSAGE is None:
        print("Missing Phone number or Message for gsm. Check the Configuration")
        return 0
    
    # actual function
    try:
        print("Sending SMS...")
        gsm = detect_gsm_module('/dev/ttyUSB2')

        for i in range(2): # retryinng for 1 time
            try:
                latitude, longitude = get_gps_location('/dev/ttyUSB1')
            except Exception as e:
                latitude, longitude = None, None
                time.sleep(1) # to retry in one second
                

        #if gsm and latitude and longitude:
        if gsm:
            location_url = f"https://maps.google.com/?q={latitude},{longitude}"
            # message = f"Driver is drowsy!\nGPS Location:\nLatitude: {latitude}\nLongitude: {longitude}\n{location_url}"
            MESSAGE += location_url
            send_sms(gsm, PHONE_NUMBER, MESSAGE)
            gsm.close()
    except Exception as e:
        print("Error sending SMS: ", e)

    # test function
    # print("SMS sending... ",PHONE_NUMBER, MESSAGE)
    # time.sleep(5)
    # print("SMS sent")

    flag_sms = 0

# Make call
def initiate_call(PHONE_NUMBER = None):
    global flag_call

    if PHONE_NUMBER is None:
        print("Missing Phone number. Check the Configuration")
        return 0
    
    # actual function
    try:
        gsm = detect_gsm_module('/dev/ttyUSB2')
        make_call(gsm, PHONE_NUMBER)
    except Exception as e:
        print("Error making call: ", e)

    # test function
    # print("Calling... ",PHONE_NUMBER)
    # time.sleep(6)
    # print("Call ended")

    flag_call = 0


# AIS-184 Warning Light Control Functions
warning_light_active = False  # Global flag for continuous warning light

def warning_light_on(duration=None):
    """Activate warning light for specified duration or continuously."""
    global flag_warning_light, warning_light_active

    if duration is None:
        # Continuous mode for failures - set flag and keep light ON
        warning_light_active = True
        print("[WARNING LIGHT] Activating in CONSTANT mode (failure)")
        # GPIO control for warning light (Linux)
        # set_warning_light_gpio_high()
        print("[TEST MODE] Warning light ON (constant) - will stay ON until cleared")
    else:
        # Timed mode for self-test or drowsiness
        print(f"[WARNING LIGHT] Activating for {duration}s")
        # set_warning_light_gpio_high()
        print(f"[TEST MODE] Warning light ON for {duration}s")
        time.sleep(duration)
        # set_warning_light_gpio_low()
        print("[TEST MODE] Warning light OFF")

    flag_warning_light = 0


def warning_light_off():
    """Deactivate warning light."""
    global warning_light_active
    warning_light_active = False
    print("[WARNING LIGHT] Deactivating")
    # set_warning_light_gpio_low()
    print("[TEST MODE] Warning light OFF")


def keep_warning_light_on():
    """
    Continuous loop to keep warning light ON during electrical failures.
    This function runs in a background thread and maintains the light state.
    Exits when warning_light_active is set to False.
    """
    global warning_light_active
    
    print("[WARNING LIGHT MONITOR] Starting continuous warning light monitor")
    
    while warning_light_active:
        # Keep GPIO high (in production, this ensures light stays on)
        # set_warning_light_gpio_high()
        
        # Log status every 10 seconds to show light is still active
        for _ in range(100):  # 10 seconds = 100 * 0.1s
            if not warning_light_active:
                break
            time.sleep(0.1)
        
        if warning_light_active:
            print("[WARNING LIGHT MONITOR] Warning light still ON (electrical failure persists)")
    
    # Turn off GPIO when exiting
    # set_warning_light_gpio_low()
    print("[WARNING LIGHT MONITOR] Warning light monitor stopped")


def buzzer_on(duration, continuous=False):
    """Activate buzzer for specified duration or continuously.
    
    Args:
        duration: Duration in seconds for each beep cycle
        continuous: If True, beep continuously until buzzer_off() is called
    """
    global buzzer_active
    
    if continuous:
        print(f"[BUZZER] Activating in CONTINUOUS mode (beeping every {duration}s)")
        buzzer_active = True
        
        # Continuous beeping loop
        while buzzer_active:
            print(f"[BUZZER] Beep cycle starting...")
            beep(file_path="high_beep.wav", decibal=5.0)
            print(f"[TEST MODE] Buzzer beep completed, waiting {duration}s before next cycle...")
            
            # Wait for next cycle, but check buzzer_active frequently to allow quick stop
            for _ in range(int(duration * 10)):
                if not buzzer_active:
                    break
                time.sleep(0.1)
        
        print("[BUZZER] Continuous mode stopped")
    else:
        # Single beep mode
        print(f"[BUZZER] Activating for {duration}s")
        beep(file_path="high_beep.wav", decibal=5.0)
        print(f"[TEST MODE] Buzzer ON for {duration}s")


def buzzer_off():
    """Stop continuous buzzer."""
    global buzzer_active
    buzzer_active = False
    print("[BUZZER] Deactivating continuous mode")


def trigger_ignition_self_test(config):
    """Trigger ignition self-test: activate warning light for 2 seconds."""
    duration = config.get('ais_184', {}).get('ignition_self_test', {}).get('warning_light_duration', 2)
    print(f"[AIS-184] Ignition self-test: activating warning light for {duration}s")
    warning_light_on(duration)
    return {"status": "SELF_TEST_COMPLETE"}


def trigger_drowsiness_warning(config):
    """Trigger drowsiness warning: warning light + buzzer."""
    print("[AIS-184] Drowsiness warning triggered")

    # Start warning light in separate thread (continuous until cleared)
    threading.Thread(target=warning_light_on, args=(None,), daemon=True).start()

    # Activate buzzer for configured duration
    buzzer_duration = config.get('ais_184', {}).get('warning_signals', {}).get('buzzer', {}).get('duration_drowsiness', 3)
    threading.Thread(target=buzzer_on, args=(buzzer_duration,)).start()


def trigger_failure_alert(failure_type, config):
    """Trigger failure alert: constant warning light + buzzer.
    
    For SENSOR_OBSTRUCTION: Continuous beeping until obstruction is removed.
    For ELECTRICAL: Single 5-second beep, but warning light stays ON continuously.
    
    Args:
        failure_type: Type of failure (ELECTRICAL or SENSOR_OBSTRUCTION)
        config: Configuration dictionary
    """
    global warning_light_active
    
    print(f"[AIS-184] Failure alert triggered: {failure_type}")

    # Activate warning light in constant mode
    warning_light_on(None)  # Set flag to True
    
    # Start continuous warning light monitor thread
    # This keeps the light ON until components are reconnected
    threading.Thread(target=keep_warning_light_on, daemon=True).start()

    # Buzzer behavior depends on failure type
    buzzer_duration = config.get('ais_184', {}).get('warning_signals', {}).get('buzzer', {}).get('duration_failure', 5)
    
    if failure_type == "SENSOR_OBSTRUCTION":
        # Continuous beeping for sensor obstruction (driver must remove obstruction)
        print(f"[AIS-184] Starting CONTINUOUS buzzer for sensor obstruction (beep every {buzzer_duration}s)")
        threading.Thread(target=buzzer_on, args=(buzzer_duration, True), daemon=True).start()
    elif failure_type == "ELECTRICAL":
        # Single beep for electrical failure, but warning light stays ON
        print(f"[AIS-184] Single buzzer activation for electrical failure ({buzzer_duration}s)")
        print(f"[AIS-184] Warning light will stay ON until components reconnected")
        threading.Thread(target=buzzer_on, args=(buzzer_duration, False)).start()


def clear_drowsiness_warning():
    """Clear drowsiness warning: deactivate warning light."""
    print("[AIS-184] Clearing drowsiness warning")
    warning_light_off()


def clear_failure_alert():
    """Clear failure alert: deactivate warning light and stop continuous buzzer."""
    print("[AIS-184] Clearing failure alert")
    warning_light_off()
    buzzer_off()
    warning_light_off()
    buzzer_off()  # Stop continuous buzzer if running


def trigger_beep_only(label):
    """Trigger a single beep for periodic high-alert repetition (no SMS/call)."""
    global flag_beep, overlap_flag_beep
    global audio_file_path_eyes_closed, audio_file_path_yawn, audio_file_path_distraction
    global audio_file_path_phone, audio_file_path_smoking

    audio_map = {
        "eyes closed": audio_file_path_eyes_closed,
        "yawn":        audio_file_path_yawn,
        "distraction": audio_file_path_distraction,
        "phone":       audio_file_path_phone,
        "smoking":     audio_file_path_smoking,
    }
    audio_file_path = audio_map.get(label, audio_file_path_eyes_closed)

    if (overlap_flag_beep == 0 and flag_beep == 0) or overlap_flag_beep == 1:
        flag_beep = 1
        print(f"[PERIODIC BEEP] Repeating beep for {label}")
        threading.Thread(target=beep, args=(audio_file_path, 3.0)).start()


# main function
def trigger(alert_type, label):
    global overlap_flag_beep, overlap_flag_led, overlap_flag_sms, overlap_flag_call
    global flag_beep, flag_led, flag_sms, flag_call
    global audio_file_path_yawn, audio_file_path_eyes_closed, audio_file_path_distraction, audio_file_path_phone, audio_file_path_smoking
    global led_duration_yawn, led_duration_eyes_closed, led_duration_distraction, led_duration_phone, led_duration_smoking  

    # make decision based on the parameter
    if alert_type is None or label is None:
       print("Alert failed: Type and label are not None")

    print("Alert type: ", alert_type)
    print("Alert label: ", label)

    actions_to_perform = []
    audio_file_path = None
    led_duration = None
    # read the list
    if label == "yawn":
        actions_to_perform = config_action_yawn.get(alert_type, [])
        audio_file_path = audio_file_path_yawn
        led_duration = led_duration_yawn
    elif label == "eyes closed":
        actions_to_perform = config_action_eyes_closed.get(alert_type, [])
        audio_file_path = audio_file_path_eyes_closed
        led_duration = led_duration_eyes_closed
    elif label == "distraction":
        actions_to_perform = config_action_distraction.get(alert_type, [])
        audio_file_path = audio_file_path_distraction
        led_duration = led_duration_distraction
    elif label == "phone":
        actions_to_perform = config_action_phone.get(alert_type, [])
        audio_file_path = audio_file_path_phone
        led_duration = led_duration_phone
    elif label == "smoking":
        actions_to_perform = config_action_smoking.get(alert_type, [])
        audio_file_path = audio_file_path_smoking
        led_duration = led_duration_smoking
    else:
        print("Invalid label. Please check the configuration file.")
        return 0
    
    # use for loop to iterate
    # make a call for each function in a list using threading
    for action in actions_to_perform:
        # sms or call function
        if isinstance(action, dict): 
            print("gsm called")
            gsm_function = ""
            if action.get("sms") != None:
                action = action.get("sms")
                gsm_function = "sms"
            elif action.get("call") != None:    
                action = action.get("call")
                gsm_function = "call"
            else:
                print("Invalid action in the configuration file.")
            
            if gsm_function == "":
                print("Invalid action in the configuration file.")

            if gsm_function == "sms":
                PHONE_NUMBER = action.get("phone_number")
                MESSAGE = action.get("message")
                if PHONE_NUMBER is None or MESSAGE is None:
                    print("Missing Phone number or Message for gsm. Check the Configuration")
                    continue
                else:
                    if (overlap_flag_sms == 0 and flag_sms == 0) or overlap_flag_sms == 1:
                        flag_sms = 1
                        threading.Thread(target=initiate_sms, args=(PHONE_NUMBER, MESSAGE,)).start()

            elif gsm_function == "call":
                PHONE_NUMBER = action.get("phone_number")
                if PHONE_NUMBER is None:
                    print("Missing Phone number. Check the Configuration")
                    continue
                else:
                    if (overlap_flag_call == 0 and flag_call == 0) or overlap_flag_call == 1:
                        flag_call = 1
                        print("Making call")
                        threading.Thread(target=initiate_call, args=(PHONE_NUMBER,)).start()
        
        elif action == "beep":
            print("Beep called")
            if (overlap_flag_beep == 0 and flag_beep == 0) or overlap_flag_beep == 1:
                print("Entered beep condition")
                flag_beep = 1
                if alert_type == "low":
                    print("Entered low beep condition")
                    threading.Thread(target=beep, args=(audio_file_path,3.0)).start()
                elif alert_type == "medium":
                    print("Entered medium beep condition")
                    threading.Thread(target=beep, args=(audio_file_path,5.0)).start()
                elif alert_type == "high":
                    print("Entered high beep condition")
                    threading.Thread(target=beep, args=(audio_file_path,3.0)).start()

        elif action == "led":
            print("Led called")
            if (overlap_flag_led == 0 and flag_led == 0) or overlap_flag_led == 1:
                flag_led = 1
                threading.Thread(target=led, args=(led_duration,)).start()
        else:
            print(f"Invalid action: {action}")




