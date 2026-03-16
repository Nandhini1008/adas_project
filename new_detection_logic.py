import yaml
from datetime import datetime
import time
from new_trigger_alert import trigger, get_gps_location
import threading
import time
import json


# Global StateBuffer class for temporal smoothing
class GlobalStateBuffer:
    """Maintains a single sliding window of ALL detection frames for temporal smoothing."""
    
    def __init__(self, window_size, awake_threshold, drowsy_threshold):
        """
        Initialize the global state buffer.
        
        Args:
            window_size: Maximum number of frames to keep in the buffer
            awake_threshold: Number of consecutive "awake" frames needed to transition to awake
            drowsy_threshold: Number of consecutive drowsy frames needed to transition to drowsy state
        """
        self.window = []
        self.window_size = window_size
        self.awake_threshold = awake_threshold
        self.drowsy_threshold = drowsy_threshold
        self.current_state = None
    
    def add_frame(self, state):
        """
        Add a new detection state to the global buffer and maintain FIFO window size.
        
        Args:
            state: The detection state to add (e.g., "awake", "eyes_closed", "yawn")
        """
        self.window.append(state)
        # Maintain window size (FIFO - remove oldest if exceeds size)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def get_smoothed_state(self):
        """
        Return the smoothed state based on consecutive frame counts.
        
        Returns:
            The smoothed state after applying temporal filtering
        """
        if not self.window:
            return self.current_state
        
        # Get the most recent state
        recent_state = self.window[-1]
        consecutive_count = 1
        
        # Count backwards to find consecutive occurrences of the recent state
        for i in range(len(self.window) - 2, -1, -1):
            if self.window[i] == recent_state:
                consecutive_count += 1
            else:
                break
        
        # Determine if we should transition based on the state type and current state
        if recent_state == "awake":
            # Transitioning TO awake: Need awake_threshold consecutive "awake" frames
            if consecutive_count >= self.awake_threshold:
                self.current_state = "awake"
                return "awake"
            else:
                # Not enough consecutive awake frames - maintain previous state
                return self.current_state if self.current_state is not None else recent_state
        else:
            # recent_state is a drowsy state (yawn, eyes_closed, distracted, phone, smoking)
            
            # Check if we're currently in "awake" state
            if self.current_state == "awake":
                # Transitioning FROM awake TO drowsy: Need drowsy_threshold consecutive frames
                if consecutive_count >= self.drowsy_threshold:
                    self.current_state = recent_state
                    return recent_state
                else:
                    # Not enough consecutive drowsy frames - stay in awake
                    return "awake"
            else:
                # Already in a drowsy state or no previous state
                # Transitioning between drowsy states: Need drowsy_threshold consecutive frames
                if consecutive_count >= self.drowsy_threshold:
                    self.current_state = recent_state
                    return recent_state
                else:
                    # Not enough consecutive frames - maintain previous state
                    return self.current_state if self.current_state is not None else recent_state


class AIS184StateManager:
    """Manages AIS-184 compliance state machine and event tracking."""

    def __init__(self, config):
        self.config = config
        self.state = "IGNITION_SELF_TEST"  # Initial state
        self.ignition_on_time = None
        self.activation_criteria_met_time = None
        self.active_monitoring_start_time = None
        self.self_test_completed = False
        self.failure_state = None  # None, "ELECTRICAL", "SENSOR_OBSTRUCTION"
        self.speed_below_threshold_time = None

    def handle_ignition_on(self):
        """Handle ignition ON event - trigger self-test."""
        self.ignition_on_time = datetime.now()
        self.state = "IGNITION_SELF_TEST"
        return {"action": "TRIGGER_SELF_TEST"}

    def handle_self_test_complete(self):
        """Mark self-test as complete and transition to inactive."""
        self.self_test_completed = True
        self.state = "INACTIVE"

    def check_activation_criteria(self, vehicle_speed):
        """Check if activation criteria are met based on speed."""
        # Get vehicle category and determine threshold
        activation_config = self.config.get('ais_184', {}).get('activation', {})
        vehicle_category = activation_config.get('vehicle_category', None)
        
        # Use category-specific threshold if vehicle_category is set
        if vehicle_category and 'category_thresholds' in activation_config:
            threshold = activation_config['category_thresholds'].get(vehicle_category, 65)
        else:
            # Fall back to legacy speed_threshold_kmh for backward compatibility
            threshold = activation_config.get('speed_threshold_kmh', 65)

        if vehicle_speed > threshold:
            if self.activation_criteria_met_time is None:
                self.activation_criteria_met_time = datetime.now()

            # Reset speed_below_threshold_time when speed goes back above threshold
            if self.speed_below_threshold_time is not None:
                self.speed_below_threshold_time = None
                print(f"[AIS-184] Speed back above threshold ({vehicle_speed:.1f} > {threshold}) - canceling deactivation timer")

            # Check if we should transition to active monitoring
            # Allow activation from INACTIVE state only.
            # FAILURE_ALERT state must be resolved explicitly by failure detection logic
            # before monitoring can resume — do NOT auto-clear failure state here.
            if self.state == "INACTIVE":
                self.state = "ACTIVE_MONITORING"
                self.active_monitoring_start_time = datetime.now()
                
                # Log vehicle category and applied threshold on activation
                log_ais_184_event(
                    "ACTIVATION_CRITERIA_MET",
                    {
                        "vehicle_speed": vehicle_speed,
                        "vehicle_category": vehicle_category if vehicle_category else "not_specified",
                        "speed_threshold": threshold
                    },
                    self.config
                )
                
                return {"action": "START_MONITORING"}
        else:
            # Speed below threshold
            if self.state == "ACTIVE_MONITORING":
                if self.speed_below_threshold_time is None:
                    self.speed_below_threshold_time = datetime.now()
                    print(f"[AIS-184] Speed below threshold ({vehicle_speed:.1f} < {threshold}) - starting deactivation timer (30s)")
                else:
                    delay = self.config.get('ais_184', {}).get('activation', {}).get('deactivation_delay', 30)
                    elapsed = (datetime.now() - self.speed_below_threshold_time).total_seconds()
                    if elapsed > delay:
                        self.state = "INACTIVE"
                        self.activation_criteria_met_time = None
                        self.speed_below_threshold_time = None
                        print(f"[AIS-184] Deactivation timer expired ({elapsed:.1f}s > {delay}s) - stopping monitoring")
                        return {"action": "STOP_MONITORING"}
                    # Log progress every 5 seconds
                    elif int(elapsed) % 5 == 0 and int(elapsed) > 0:
                        remaining = delay - elapsed
                        print(f"[AIS-184] Deactivation in {remaining:.0f}s (speed: {vehicle_speed:.1f} km/h)")
            else:
                self.speed_below_threshold_time = None
                self.activation_criteria_met_time = None

        return None


    def get_activation_delay(self):
        """Calculate delay between activation criteria and active monitoring."""
        if self.activation_criteria_met_time and self.active_monitoring_start_time:
            return (self.active_monitoring_start_time - self.activation_criteria_met_time).total_seconds()
        return None

    def set_failure_state(self, failure_type):
        """Set failure state (ELECTRICAL or SENSOR_OBSTRUCTION)."""
        self.failure_state = failure_type
        self.state = "FAILURE_ALERT"

    def clear_failure_state(self):
        """Clear failure state and return to active monitoring."""
        self.failure_state = None
        if self.active_monitoring_start_time:
            self.state = "ACTIVE_MONITORING"
        else:
            self.state = "INACTIVE"


class FailureDetector:
    """Detects electrical failures and sensor obstructions."""

    def __init__(self, config):
        self.config = config
        self.last_detection_time = None
        self.low_quality_frame_count = 0
        self.electrical_failure_detected = False
        self.sensor_obstruction_detected = False
        self.obstruction_confirmed_time = None  # when obstruction was first confirmed active
        self.monitoring_active_since = None     # set when active monitoring starts; used for warm-up grace period

    def check_electrical_failure(self, current_time):
        """Check for electrical failure based on detection timeout."""
        if not self.config.get('ais_184', {}).get('failure_detection', {}).get('electrical', {}).get('enabled', False):
            return False

        timeout = self.config.get('ais_184', {}).get('failure_detection', {}).get('electrical', {}).get('detection_timeout', 1)

        if self.last_detection_time is None:
            self.last_detection_time = current_time
            return False

        time_since_last = (current_time - self.last_detection_time).total_seconds()

        if time_since_last > timeout:
            self.electrical_failure_detected = True
            return True

        return False

    def update_detection_heartbeat(self):
        """Update heartbeat to indicate system is functioning."""
        self.last_detection_time = datetime.now()
        if self.electrical_failure_detected:
            self.electrical_failure_detected = False
            return {"action": "ELECTRICAL_FAILURE_RESOLVED"}
        return None

    def check_sensor_obstruction(self, detection_confidence):
        """Check for sensor obstruction based on detection quality."""
        if not self.config.get('ais_184', {}).get('failure_detection', {}).get('sensor_obstruction', {}).get('enabled', False):
            return False

        threshold = self.config.get('ais_184', {}).get('failure_detection', {}).get('sensor_obstruction', {}).get('quality_threshold', 0.3)
        required_frames = self.config.get('ais_184', {}).get('failure_detection', {}).get('sensor_obstruction', {}).get('consecutive_frames', 20)

        if detection_confidence < threshold:
            self.low_quality_frame_count += 1
        else:
            if self.sensor_obstruction_detected:
                # Quality restored
                self.sensor_obstruction_detected = False
                self.low_quality_frame_count = 0
                return {"action": "SENSOR_OBSTRUCTION_RESOLVED"}
            self.low_quality_frame_count = 0

        if self.low_quality_frame_count >= required_frames and not self.sensor_obstruction_detected:
            self.sensor_obstruction_detected = True
            return {"action": "SENSOR_OBSTRUCTION_DETECTED"}

        return None


def log_ais_184_event(event_type, details, config):
    """Log AIS-184 compliance events to dedicated log file."""
    if not config.get('ais_184', {}).get('logging', {}).get('enabled', False):
        return

    log_file = config.get('ais_184', {}).get('logging', {}).get('log_file', 'ais_184_compliance_log.txt')
    timestamp = datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details
    }

    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[AIS-184] Error logging event: {e}")


def simulate_ignition_on():
    """
    Simulate ignition ON event for testing (reads from ignition_state.txt).
    Tracks previous state to detect OFF → ON transitions.
    
    Returns:
        tuple: (is_on, is_transition) where:
            - is_on: True if ignition is currently ON
            - is_transition: True if this is an OFF → ON transition
    """
    try:
        # Read current ignition state
        with open("ignition_state.txt", "r") as f:
            current_state = f.read().strip().lower()
            is_on = current_state == "on"
        
        # Read previous ignition state
        try:
            with open("ignition_state_previous.txt", "r") as f:
                previous_state = f.read().strip().lower()
                was_on = previous_state == "on"
        except FileNotFoundError:
            # First run - no previous state
            was_on = False
        
        # Detect OFF → ON transition
        is_transition = (not was_on) and is_on
        
        # Store current state as previous for next iteration
        with open("ignition_state_previous.txt", "w") as f:
            f.write(current_state)
        
        return is_on, is_transition
        
    except FileNotFoundError:
        # Default to ON for normal operation, but no transition
        return True, False


def get_detection_confidence():
    """Extract detection confidence from log_detection_confidence.txt for quality assessment."""
    try:
        with open("log_detection_confidence.txt", "r") as f:
            confidence = float(f.read().strip())
            return confidence
    except (FileNotFoundError, ValueError):
        # Default to high confidence if file doesn't exist
        return 1.0


# Global StateBuffer class for temporal smoothing
class GlobalStateBuffer:
    """Maintains a single sliding window of ALL detection frames for temporal smoothing."""
    
    def __init__(self, window_size, awake_threshold, drowsy_threshold):
        """
        Initialize the global state buffer.
        
        Args:
            window_size: Maximum number of frames to keep in the buffer
            awake_threshold: Number of consecutive "awake" frames needed to transition to awake
            drowsy_threshold: Number of consecutive drowsy frames needed to transition to drowsy state
        """
        self.window = []
        self.window_size = window_size
        self.awake_threshold = awake_threshold
        self.drowsy_threshold = drowsy_threshold
        self.current_state = None
    
    def add_frame(self, state):
        """
        Add a new detection state to the global buffer and maintain FIFO window size.
        
        Args:
            state: The detection state to add (e.g., "awake", "eyes_closed", "yawn")
        """
        self.window.append(state)
        # Maintain window size (FIFO - remove oldest if exceeds size)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def get_smoothed_state(self):
        """
        Return the smoothed state based on consecutive frame counts.
        
        Returns:
            The smoothed state after applying temporal filtering
        """
        if not self.window:
            return self.current_state
        
        # Get the most recent state
        recent_state = self.window[-1]
        consecutive_count = 1
        
        # Count backwards to find consecutive occurrences of the recent state
        for i in range(len(self.window) - 2, -1, -1):
            if self.window[i] == recent_state:
                consecutive_count += 1
            else:
                break
        
        # Determine if we should transition based on the state type and current state
        if recent_state == "awake":
            # Transitioning TO awake: Need awake_threshold consecutive "awake" frames
            if consecutive_count >= self.awake_threshold:
                self.current_state = "awake"
                return "awake"
            else:
                # Not enough consecutive awake frames - maintain previous state
                return self.current_state if self.current_state is not None else recent_state
        else:
            # recent_state is a drowsy state (yawn, eyes_closed, distracted, phone, smoking)
            
            # Check if we're currently in "awake" state
            if self.current_state == "awake":
                # Transitioning FROM awake TO drowsy: Need drowsy_threshold consecutive frames
                if consecutive_count >= self.drowsy_threshold:
                    self.current_state = recent_state
                    return recent_state
                else:
                    # Not enough consecutive drowsy frames - stay in awake
                    return "awake"
            else:
                # Already in a drowsy state or no previous state
                # Transitioning between drowsy states: Need drowsy_threshold consecutive frames
                if consecutive_count >= self.drowsy_threshold:
                    self.current_state = recent_state
                    return recent_state
                else:
                    # Not enough consecutive frames - maintain previous state
                    return self.current_state if self.current_state is not None else recent_state


# load configuration
with open("new_config.yaml","r",encoding="utf-8") as file:
    config = yaml.safe_load(file)

flow = config.get('flow', []) # get the flow configuration
config = config.get('config', {}) # get the configuration

# To check the final flow is declared correct or not in config file
if len(flow) != 3:
    print("Invalid flow configuration. The flow value should be a list of three elements.")
    exit()

print("Configuration Verified")

vehicle_speed_config = config.get('vehicle_speed_threshold') 
current_config = {} # current configuration 
detection_config = {}  # detection configuration
current_config = config.get(flow[0], {}) # get the current configuration
detection_config = current_config.get(flow[1], {}) # get the detection configuration
# Confiuration Timers
config_timers = detection_config.get('timer', {})
config_awake_timer = config_timers.get('awake')
config_yawn_timer = config_timers.get('yawn')
config_eyes_closed_timer = config_timers.get('eyes_closed')
config_distraction_timer = config_timers.get('distraction')
config_phone_timer = config_timers.get('phone')
config_smoking_timer = config_timers.get('smoking')

# Alert Timers
config_low_alert_timer = detection_config.get('low_alert_timer', {})
config_medium_alert_timer = detection_config.get('medium_alert_timer', {})
config_high_alert_timer = detection_config.get('high_alert_timer', {})

# Thresholds
config_thresholds = detection_config.get('threshold_percentage', {})

# Time sleep
config_time_sleep = detection_config.get('time_sleep', {}).get('time_sleep')

# Load temporal smoothing configuration
temporal_smoothing_config = config.get('temporal_smoothing', {})
temporal_smoothing_enabled = temporal_smoothing_config.get('enabled', False)
temporal_window_size = temporal_smoothing_config.get('window_size', 5)
temporal_awake_threshold = temporal_smoothing_config.get('consecutive_awake_threshold', 3)
temporal_drowsy_threshold = temporal_smoothing_config.get('consecutive_drowsy_threshold', 1)

# Validate temporal smoothing configuration
if temporal_smoothing_enabled:
    if not (1 <= temporal_awake_threshold <= temporal_window_size):
        print(f"Invalid temporal_smoothing configuration: consecutive_awake_threshold ({temporal_awake_threshold}) must be between 1 and window_size ({temporal_window_size})")
        exit()
    if not (1 <= temporal_drowsy_threshold <= temporal_window_size):
        print(f"Invalid temporal_smoothing configuration: consecutive_drowsy_threshold ({temporal_drowsy_threshold}) must be between 1 and window_size ({temporal_window_size})")
        exit()

# counter variables
counter_awake = 0
counter_yawn = 0
counter_eye_closed = 0
counter_distraction = 0
counter_phone = 0
counter_smoking = 0

# timer variables
timer_awake = datetime.now()
timer_yawn = datetime.now()
timer_eye_closed = datetime.now()
timer_distraction = datetime.now()
timer_phone = datetime.now()
timer_smoking = datetime.now()

# flag variable to check the timer is started or not
flag_yawn = 0
flag_eye_closed = 0 
flag_distraction = 0
flag_phone = 0
flag_smoking = 0 

# timer for to send alert
alert_timer_yawn = datetime.now()
alert_timer_eye_closed = datetime.now()
alert_timer_distraction = datetime.now()
alert_timer_phone = datetime.now()
alert_timer_smoking = datetime.now()

# timer for periodic high alert (repeat every 3 seconds while in high alert)
last_high_alert_yawn = None
last_high_alert_eye_closed = None
last_high_alert_distraction = None
last_high_alert_phone = None
last_high_alert_smoking = None
high_alert_repeat_interval = 3  # seconds

# alert type - [low, medium, high]
alert_type_yawn = [0,0,0]
alert_type_eye_closed = [0,0,0]
alert_type_distraction = [0,0,0]
alert_type_phone = [0,0,0]
alert_type_smoking = [0,0,0]

# alert variables for beep function
alert_phone = 0
alert_smoking = 0
traffic_info = 12 # randomly assigned / need to integrate input function for traffic
eye_closed_state = 0 # 0 - eye open, 1 - eye closed
yawn_count_alert = 0 # taking yawn count for drowsy detection
isTriggerEyeClosed = False # to trigger eye closed alert
isCallDriver = False # to call driver
isSmstoOwner = False # to send sms to owner

# time_sleep = config["time_sleep"] # sleep time to read the logs 1 


def get_smoothed_state(raw_status, global_buffer, temporal_config):
    """
    Apply temporal smoothing to the raw detection status using a single global buffer.
    
    Args:
        raw_status: The raw detection status from log_detection.txt
        global_buffer: Single GlobalStateBuffer instance tracking all detections
        temporal_config: Dictionary with temporal smoothing configuration
        
    Returns:
        The smoothed status after applying temporal filtering
    """
    # Check if temporal smoothing is enabled
    if not temporal_config.get('enabled', False):
        return raw_status
    
    # Add the raw status to the global buffer
    global_buffer.add_frame(raw_status)
    
    # Get the smoothed state from the buffer
    smoothed_state = global_buffer.get_smoothed_state()
    
    return smoothed_state if smoothed_state is not None else raw_status


# function to count the timer
def count_timer(timer):
    timer = datetime.now() - timer
    return timer

# function to trigger beep [not implemented and tested]
def alert(label,counter,timer,frequency):
    global alert_phone, alert_smoking, traffic_info, eye_closed_state, yawn_count_alert,isTriggerEyeClosed,isCallDriver,isSmstoOwner

    if label=="phone usage":
        if alert_phone == 0:
            for _ in range(counter):
                # triggger beep need to added
                print("trigger beep")
                time.sleep(1)
        elif alert_phone == 1:
            # msg to send using gsm to be added
            print("Send message to owner")
        else:
            # need to discuss
            pass
        alert_phone += 1

    if label=="smoke detected":
        if alert_smoking == 0:
            for _ in range(counter):
                # triggger beep need to added
                print("trigger beep")
                time.sleep(1)
        elif alert_smoking == 1:
            # msg to send using gsm to be added
            print("Send message to owner")
        else:
            # need to discuss
            pass
        alert_smoking += 1
        
    if label=="distracted": # confusion - when to stop the beep
        if traffic_info < 40 : # have to integrate traffic info logic/function
            #low beep
            print(counter_distraction,timer_distraction,frequency) # low beep on high traffic
        else :
            #high beep
            print(counter_distraction,timer_distraction,frequency) # high beep on low traffic

    if label=="yawn": # confusion - what to do when eye closed state
        if eye_closed_state == 1: 
            print("trigger beep on the frequency",frequency)
        else:
            pass # to be discussed
        yawn_count_alert += 1

    if label=="eye closed": # for drowsy state
        if yawn_count_alert > 5: # based on yamn count, actions are taken
            if isTriggerEyeClosed == False:
                print("trigger beep on the frequency",frequency)
                isTriggerEyeClosed = True
            elif isCallDriver == False:
                print("call to driver")
                isCallDriver = True
            elif isSmstoOwner == False:
                print("sms to owner")   
                isSmstoOwner = True
        else: # for normal eye closed state, not including yawn
            for _ in range(counter):
                # triggger beep need to added
                print("trigger beep")
                time.sleep(1)

    """
    logics to clarify:
     - when to stop the beep
     - when to reset the values of alert_phone, alert_smoking, traffic_info, eye_closed_state, yawn_count_alert,isTriggerEyeClosed,isCallDriver,isSmstoOwner
     - In distraction, what kind of inputs are taken
    """

def count_percentage(time_sleep, alert_timer,counter):
    total_counter = alert_timer * (100/(time_sleep*100)) # calculate the total counter in given seconds
    return (counter/total_counter)*100

# function to update the status od timer and counter
def update_status(status, counter, timer, config_timer, flag, alert_timer, alert_type, config_low_timer, config_medium_timer, config_high_timer, config_percentage, last_high_alert):
    global config_time_sleep, high_alert_repeat_interval
    counter += 1
    if counter == 1:  # Initiate the timer
        timer = datetime.now()
    #print(int(counter*time_sleep))                                 
    if count_timer(timer).total_seconds() < config_timer:
        if flag == 0: # Start timer for triggering different types of alert
            alert_timer = datetime.now()
            flag = 1
    
        # low alert
        if count_timer(alert_timer).total_seconds() <= config_low_timer and (alert_type[0] == 0) and (count_percentage(config_time_sleep, config_low_timer,counter) >= config_percentage):
            print("Low alert for ",status)
            trigger("low",status)

            alert_type[0] = 1
            timer = datetime.now()
        # medium alert
        elif count_timer(alert_timer).total_seconds() <= config_medium_timer and count_timer(alert_timer).total_seconds() > config_low_timer and (alert_type[0] == 1) and (alert_type[1] == 0) and (count_percentage(config_time_sleep, config_medium_timer,counter) >= config_percentage):
            print("Medium alert for ",status)
            trigger("medium",status)

            alert_type[1] = 1
            timer = datetime.now()
        elif count_timer(alert_timer).total_seconds() <= config_high_timer and count_timer(alert_timer).total_seconds() > config_medium_timer and (alert_type[0] == 1) and (alert_type[1] == 1) and (alert_type[2] == 0) and (count_percentage(config_time_sleep, config_high_timer,counter) >= config_percentage):
            print("High alert for ",status)
            trigger("high",status) 

            alert_type[2] = 1
            last_high_alert = datetime.now()  # Record when high alert was triggered
            # DO NOT reset after high alert - stay in high alert state
            # Counter continues incrementing, alert state remains [1,1,1]
            # Only reset when state actually changes (handled by temporal smoothing)
            timer = datetime.now()
        # Periodic high alert: if in high alert state, repeat BEEP ONLY every N seconds
        elif alert_type[2] == 1 and last_high_alert is not None:
            time_since_last_high_alert = count_timer(last_high_alert).total_seconds()
            if time_since_last_high_alert >= high_alert_repeat_interval:
                print(f"[PERIODIC HIGH ALERT] Repeating beep for {status} (every {high_alert_repeat_interval}s)")
                from new_trigger_alert import trigger_beep_only
                trigger_beep_only(status)
                last_high_alert = datetime.now()  # Reset the timer for next periodic alert
        elif (count_timer(alert_timer).total_seconds() > config_high_timer and (alert_type[2] == 0)) or (count_timer(alert_timer).total_seconds() > config_medium_timer and (alert_type[1] == 0)) or ((count_timer(alert_timer).total_seconds() > config_low_timer) and (alert_type[0] == 0)):
            print("Resetting values for ",status)
            # resetting values
            counter = 0
            #timer = datetime.now()
            flag = 0
            alert_type = [0,0,0]
            alert_timer = datetime.now()
            last_high_alert = None  # Reset periodic alert timer
            
    if count_timer(timer).total_seconds() > config_timer: # time exceeded 
        print("Time exceeded: Resetting values for ",status)
        # resetting
        counter = 0
        #timer = datetime.now()
        flag = 0
        alert_type = [0,0,0]
        alert_timer = datetime.now()
        last_high_alert = None  # Reset periodic alert timer

    return counter, timer, flag, alert_type, alert_timer, last_high_alert # to update the counter and timer values globally

# main function to read the log and perform detection logic
def run_detection_logic():
    global counter_awake, counter_yawn, counter_eye_closed, counter_distraction, counter_phone, counter_smoking
    global timer_awake, timer_yawn, timer_eye_closed, timer_distraction, timer_phone, timer_smoking 
    global flag_distraction, flag_phone, flag_smoking, flag_yawn, flag_eye_closed
    global alert_timer_yawn, alert_timer_eye_closed, alert_timer_distraction, alert_timer_phone, alert_timer_smoking
    global alert_type_yawn, alert_type_eye_closed, alert_type_distraction, alert_type_phone, alert_type_smoking
    global config_time_sleep
    global last_high_alert_yawn, last_high_alert_eye_closed, last_high_alert_distraction, last_high_alert_phone, last_high_alert_smoking
    
    # Initialize SINGLE global state buffer for temporal smoothing
    global_buffer = None
    if temporal_smoothing_enabled:
        global_buffer = GlobalStateBuffer(
            temporal_window_size, 
            temporal_awake_threshold, 
            temporal_drowsy_threshold
        )
    
    # Temporal smoothing configuration dictionary
    temporal_config = {
        'enabled': temporal_smoothing_enabled,
        'window_size': temporal_window_size,
        'consecutive_awake_threshold': temporal_awake_threshold,
        'consecutive_drowsy_threshold': temporal_drowsy_threshold
    }
    
    # Track previous state to detect genuine state changes
    previous_smoothed_status = None
    
    # No-detection counter: tracks consecutive "awake" (no face/no detection) frames
    # After NO_DETECT_THRESHOLD consecutive no-detection frames, trigger continuous beep
    NO_DETECT_THRESHOLD = 5
    no_detection_count = 0
    no_detection_alert_active = False
    
    # ── Startup cleanup ───────────────────────────────────────────────────────
    # Clear stale failure state from previous runs so the overlay starts clean.
    try:
        with open("log_detection.txt", "w") as _f:
            _f.write("awake")
        with open("log_detection_confidence.txt", "w") as _f:
            _f.write("1.0")
        # Clear stale compliance log so old SENSOR_OBSTRUCTION doesn't show at startup
        with open("ais_184_compliance_log.txt", "w") as _f:
            _f.write("")
    except Exception:
        pass

    # AIS-184 initialization
    ais_184_enabled = config.get('ais_184', {}).get('enabled', False)
    ais_state_manager = None
    failure_detector = None
    ais_ignition_checked = False

    if ais_184_enabled:
        print("[AIS-184] Compliance mode enabled")
        ais_state_manager = AIS184StateManager(config)
        failure_detector = FailureDetector(config)
        
        # Check ignition state and trigger self-test ONLY on OFF to ON transition
        is_ignition_on, is_transition = simulate_ignition_on()
        if is_ignition_on and is_transition:
            print("[AIS-184] Ignition OFF to ON transition detected - triggering self-test")
            log_ais_184_event("IGNITION_TRANSITION", {"from": "OFF", "to": "ON"}, config)
            result = ais_state_manager.handle_ignition_on()
            if result and result.get("action") == "TRIGGER_SELF_TEST":
                # Import trigger function from new_trigger_alert
                from new_trigger_alert import trigger_ignition_self_test
                test_result = trigger_ignition_self_test(config)
                ais_state_manager.handle_self_test_complete()
                log_ais_184_event("IGNITION_SELF_TEST", {"status": "COMPLETE"}, config)
                print("[AIS-184] Self-test completed")
        elif is_ignition_on and not is_transition:
            print("[AIS-184] Ignition already ON - skipping self-test (no transition)")
            # Skip self-test, go directly to inactive state
            ais_state_manager.self_test_completed = True
            ais_state_manager.state = "INACTIVE"
        else:
            print("[AIS-184] Ignition OFF - system inactive")
            ais_state_manager.state = "INACTIVE"
        ais_ignition_checked = True
    
    # checking given configuration ratio matching with 6:10 ratio format
    # check_configuration(config["set_counter_eye_closed"], config["timer_eye_closed"])

    while True:
        # CAN VALIDATION
        # Original: "/opt/edgeai-gst-apps/apps_python/vehicle_speed_log.txt"
        with open("vehicle_speed_log.txt", "r") as file:
            try:
                current_vehicle_speed = float(file.read().strip())
            except Exception as e:
                current_vehicle_speed = None
                print("Invalid vehicle speed from vehicle_speed_log.txt")
        
        # AIS-184: Check activation criteria based on speed
        if ais_184_enabled and ais_state_manager and current_vehicle_speed is not None:
            activation_result = ais_state_manager.check_activation_criteria(current_vehicle_speed)
            if activation_result:
                if activation_result.get("action") == "START_MONITORING":
                    print(f"[AIS-184] Active monitoring started at speed {current_vehicle_speed} km/h")
                    delay = ais_state_manager.get_activation_delay()
                    if delay is not None:
                        print(f"[AIS-184] Activation delay: {delay:.2f} seconds")
                        log_ais_184_event("ACTIVE_MONITORING_START", {"delay": delay, "speed": current_vehicle_speed}, config)
                        activation_delay_max = config.get('ais_184', {}).get('activation', {}).get('activation_delay_max', 300)
                        if delay > activation_delay_max:
                            print(f"[AIS-184] WARNING: Activation delay ({delay:.2f}s) exceeds {activation_delay_max}s limit!")
                            log_ais_184_event("ACTIVATION_DELAY_VIOLATION", {"delay": delay, "limit": activation_delay_max}, config)
                    # Flush temporal buffer so pre-activation frames don't bleed into
                    # detection counters and cause false positives immediately after activation.
                    if global_buffer:
                        global_buffer.window.clear()
                        global_buffer.current_state = None
                    previous_smoothed_status = None
                    # Reset all detection counters so nothing carries over from inactive period
                    counter_awake = counter_yawn = counter_eye_closed = 0
                    counter_distraction = counter_phone = counter_smoking = 0
                    flag_yawn = flag_eye_closed = flag_distraction = 0
                    flag_phone = flag_smoking = 0
                    alert_type_yawn = alert_type_eye_closed = [0,0,0]
                    alert_type_distraction = alert_type_phone = alert_type_smoking = [0,0,0]
                    last_high_alert_yawn = last_high_alert_eye_closed = None
                    last_high_alert_distraction = last_high_alert_phone = last_high_alert_smoking = None
                    # Reset obstruction state and warm-up timer
                    if failure_detector:
                        failure_detector.monitoring_active_since = datetime.now()
                        failure_detector.low_quality_frame_count = 0
                        failure_detector.sensor_obstruction_detected = False
                    # Reset no-detection counter
                    no_detection_count = 0
                    no_detection_alert_active = False
                    # Write neutral confidence so the ML's first real frame is used
                    try:
                        with open("log_detection_confidence.txt", "w") as _f:
                            _f.write("1.0")
                    except Exception:
                        pass
                elif activation_result.get("action") == "STOP_MONITORING":
                    print(f"[AIS-184] Active monitoring stopped (speed below threshold)")
                    log_ais_184_event("ACTIVE_MONITORING_STOP", {"speed": current_vehicle_speed}, config)
        
        # AIS-184: Failure detection (only when actively monitoring)
        if ais_184_enabled and failure_detector and ais_state_manager and ais_state_manager.state == "ACTIVE_MONITORING":
            # ── Electrical failure ────────────────────────────────────────────
            # Only check when NOT already in a confirmed electrical failure state.
            # Once confirmed, we poll log_detection.txt directly for ML recovery.
            if not failure_detector.electrical_failure_detected:
                electrical_failure = failure_detector.check_electrical_failure(datetime.now())
                if electrical_failure:
                    print("[AIS-184] ELECTRICAL FAILURE DETECTED - ML process not responding")
                    ais_state_manager.set_failure_state("ELECTRICAL")
                    log_ais_184_event("ELECTRICAL_FAILURE", {
                        "timeout": config.get('ais_184', {}).get('failure_detection', {}).get('electrical', {}).get('detection_timeout', 1),
                        "reason": "ML process timeout"
                    }, config)
                    # Write electrical failure to log files immediately so overlay can display it
                    # Include component detail (ML/Camera system)
                    with open("log_detection.txt", "w") as f:
                        f.write("electrical_failure:Camera/ML")
                    with open("log_detection_confidence.txt", "w") as f:
                        f.write("0.0")
                    from new_trigger_alert import trigger_failure_alert
                    trigger_failure_alert("ELECTRICAL", config)
            else:
                # Already in electrical failure — check if ML process has resumed
                # by reading log_detection.txt for a valid (non-failure) value.
                try:
                    with open("log_detection.txt", "r") as _f:
                        _probe = _f.read().strip()
                    if _probe not in ("electrical_failure", "sensor_obstruction", ""):
                        # ML is writing again — resolve the failure
                        failure_detector.electrical_failure_detected = False
                        failure_detector.last_detection_time = datetime.now()
                        ais_state_manager.clear_failure_state()
                        log_ais_184_event("ELECTRICAL_FAILURE_RESOLVED", {}, config)
                        print("[AIS-184] Electrical failure resolved")
                        from new_trigger_alert import clear_failure_alert
                        clear_failure_alert()
                        # Reset temporal buffer so stale failure states don't linger
                        if global_buffer:
                            global_buffer.window.clear()
                            global_buffer.current_state = None
                    else:
                        # ML still down — keep failure state in logs with component detail
                        with open("log_detection.txt", "w") as f:
                            f.write("electrical_failure:Camera/ML")
                        with open("log_detection_confidence.txt", "w") as f:
                            f.write("0.0")
                except FileNotFoundError:
                    with open("log_detection.txt", "w") as f:
                        f.write("electrical_failure:Camera/ML")
                    with open("log_detection_confidence.txt", "w") as f:
                        f.write("0.0")

            # ── Sensor obstruction ────────────────────────────────────────────
            # Only run obstruction checks when NOT in electrical failure
            if not failure_detector.electrical_failure_detected:
                if not failure_detector.sensor_obstruction_detected:
                    # Not currently in obstruction — check confidence to detect it.
                    # Skip during warm-up grace period after monitoring activation
                    # to avoid false triggers from stale confidence values.
                    warmup_sec = 3  # seconds to wait after activation before checking
                    in_warmup = (
                        failure_detector.monitoring_active_since is not None and
                        (datetime.now() - failure_detector.monitoring_active_since).total_seconds() < warmup_sec
                    )
                    if not in_warmup:
                        detection_confidence = get_detection_confidence()
                        obstruction_result = failure_detector.check_sensor_obstruction(detection_confidence)
                        if obstruction_result and obstruction_result.get("action") == "SENSOR_OBSTRUCTION_DETECTED":
                            print(f"[AIS-184] SENSOR OBSTRUCTION DETECTED (confidence: {detection_confidence})")
                            ais_state_manager.set_failure_state("SENSOR_OBSTRUCTION")
                            failure_detector.obstruction_confirmed_time = datetime.now()
                            log_ais_184_event("SENSOR_OBSTRUCTION", {"confidence": detection_confidence}, config)
                            from new_trigger_alert import trigger_failure_alert
                            trigger_failure_alert("SENSOR_OBSTRUCTION", config)
                            if global_buffer:
                                global_buffer.window.clear()
                                global_buffer.current_state = None
                            # Immediately lock the log files so the ML can't write "awake"
                            # before the next cycle's probe runs
                            with open("log_detection.txt", "w") as f:
                                f.write("sensor_obstruction")
                            with open("log_detection_confidence.txt", "w") as f:
                                f.write("0.0")
                else:
                    # Already in obstruction.
                    # Enforce a minimum hold time before allowing resolution.
                    min_hold_sec = 3
                    hold_elapsed = (datetime.now() - failure_detector.obstruction_confirmed_time).total_seconds() \
                        if failure_detector.obstruction_confirmed_time else 0

                    if hold_elapsed >= min_hold_sec:
                        # Read confidence BEFORE overwriting — if ML has resumed,
                        # it will have written a real value (> threshold) here.
                        # We do NOT overwrite log_detection_confidence.txt so the ML
                        # can write its real confidence value for us to read.
                        recovery_confidence = get_detection_confidence()
                        threshold = failure_detector.config.get('ais_184', {}).get(
                            'failure_detection', {}).get('sensor_obstruction', {}).get('quality_threshold', 0.3)

                        if recovery_confidence > threshold:
                            # ML is producing real detections — obstruction cleared
                            failure_detector.sensor_obstruction_detected = False
                            failure_detector.low_quality_frame_count = 0
                            failure_detector.obstruction_confirmed_time = None
                            ais_state_manager.clear_failure_state()
                            log_ais_184_event("SENSOR_OBSTRUCTION_RESOLVED", {}, config)
                            print(f"[AIS-184] Sensor obstruction resolved (confidence: {recovery_confidence:.2f})")
                            if global_buffer:
                                global_buffer.window.clear()
                                global_buffer.current_state = None
                            from new_trigger_alert import clear_failure_alert
                            clear_failure_alert()
                            # Don't overwrite logs — let ML take over
                        else:
                            # Still obstructed — only lock log_detection.txt
                            # Leave log_detection_confidence.txt alone so ML can write to it
                            with open("log_detection.txt", "w") as f:
                                f.write("sensor_obstruction")
                    else:
                        # Still in hold period — keep both logs locked
                        with open("log_detection.txt", "w") as f:
                            f.write("sensor_obstruction")
                        with open("log_detection_confidence.txt", "w") as f:
                            f.write("0.0")
        
        try:
            if float(vehicle_speed_config) < current_vehicle_speed: # vehicle speed is greater than the configured value
                alert_system_status = True
            elif current_vehicle_speed is None:
                print("Vehicle speed is not available. Please check the CAN connection.")
                # time.sleep(1)
                continue
            else:
                alert_system_status = False # vehicle speed is less than the configured value
        except Exception as e:
            print("Cannot read the vehicle speed from vehicle_speed_log.txt")
            continue

        # AIS-184: Update heartbeat timestamp when system is running normally
        # (not during electrical failure — that's handled above via file probe)
        if ais_184_enabled and failure_detector and not failure_detector.electrical_failure_detected:
            failure_detector.last_detection_time = datetime.now()

        if alert_system_status == True:
            # read the status from log_detection.txt
            print(counter_awake, counter_yawn, counter_eye_closed, counter_distraction, counter_phone, counter_smoking," Vehicle speed config: ",vehicle_speed_config," Current Vehicle Speed: ", current_vehicle_speed)
            # Original: "/opt/edgeai-gst-apps/apps_python/log_detection.txt"
            with open("log_detection.txt","r") as file:
                status = file.read().strip()

            # ── Immediate electrical failure detection ────────────────────────────────────
            # If log_detection.txt contains "electrical_failure", trigger failure immediately.
            # This handles pre-startup failures (from check_electrical_system.py) and
            # manual testing scenarios.
            if status.startswith("electrical_failure") and ais_184_enabled and ais_state_manager:
                if not failure_detector.electrical_failure_detected:
                    # Extract component details if present
                    component = "Unknown"
                    if ":" in status:
                        component = status.split(":", 1)[1]
                    
                    print(f"[AIS-184] ELECTRICAL FAILURE DETECTED from log file - Component: {component}")
                    failure_detector.electrical_failure_detected = True
                    ais_state_manager.set_failure_state("ELECTRICAL")
                    log_ais_184_event("ELECTRICAL_FAILURE", {
                        "reason": "log_file_indicator",
                        "component": component
                    }, config)
                    from new_trigger_alert import trigger_failure_alert
                    trigger_failure_alert("ELECTRICAL", config)
                    # Continue to skip further detection logic
                    time.sleep(config_time_sleep)
                    continue

            # ── No-detection tracking (only when actively monitoring OR in sensor obstruction failure) ─────────
            # The ML writes "no_detection" when it finds no face in the frame.
            # After NO_DETECT_THRESHOLD consecutive such frames, treat as sensor
            # obstruction and start continuous beep.
            # Recovery: the FIRST frame with a real detection clears everything immediately.
            # IMPORTANT: Recovery must work even when in FAILURE_ALERT state, so check for
            # ACTIVE_MONITORING *or* sensor obstruction failure state.
            is_active = (not ais_184_enabled) or (
                ais_state_manager and (
                    ais_state_manager.state == "ACTIVE_MONITORING" or
                    (ais_state_manager.state == "FAILURE_ALERT" and 
                     ais_state_manager.failure_state == "SENSOR_OBSTRUCTION")
                )
            )
            if is_active:
                if status not in ("no_detection", "sensor_obstruction", "electrical_failure", ""):
                    # Real detection (awake, eyes_closed, distracted, etc.) — reset counter
                    no_detection_count = 0
                    if no_detection_alert_active:
                        no_detection_alert_active = False
                        print("[DETECT-LOGIC] No-detection alert cleared immediately (face detected)")
                        from new_trigger_alert import buzzer_off, clear_failure_alert
                        buzzer_off()
                        clear_failure_alert()
                        if ais_184_enabled and ais_state_manager:
                            ais_state_manager.clear_failure_state()
                            log_ais_184_event("SENSOR_OBSTRUCTION_RESOLVED", {"reason": "face_detected"}, config)
                        if global_buffer:
                            global_buffer.window.clear()
                            global_buffer.current_state = None
                        # Don't overwrite status — let it pass through to detection logic
                        # so counters can resume from the real detection state
                else:
                    if status == "no_detection":
                        no_detection_count += 1

                if no_detection_count >= NO_DETECT_THRESHOLD and not no_detection_alert_active:
                    no_detection_alert_active = True
                    print(f"[DETECT-LOGIC] {NO_DETECT_THRESHOLD} consecutive no-detections — starting continuous beep")
                    with open("log_detection.txt", "w") as _f:
                        _f.write("sensor_obstruction")
                    with open("log_detection_confidence.txt", "w") as _f:
                        _f.write("0.0")
                    from new_trigger_alert import buzzer_on
                    import threading
                    threading.Thread(target=buzzer_on, args=(5, True), daemon=True).start()
                    if ais_184_enabled and ais_state_manager:
                        ais_state_manager.set_failure_state("SENSOR_OBSTRUCTION")
                        log_ais_184_event("SENSOR_OBSTRUCTION", {"reason": "no_detection", "frames": no_detection_count}, config)

            # Skip further detection logic when in any failure state
            if ais_184_enabled and ais_state_manager and ais_state_manager.failure_state is not None:
                time.sleep(config_time_sleep)
                continue
            
            # Apply temporal smoothing to the raw status
            smoothed_status = get_smoothed_state(status, global_buffer, temporal_config)
            
            # Optional: Log raw vs smoothed status for debugging
            if temporal_smoothing_enabled and status != smoothed_status:
                print(f"[TEMPORAL SMOOTHING] Raw: {status} -> Smoothed: {smoothed_status}")
            
            # Detect genuine state changes and reset counters for other detection types
            if previous_smoothed_status is not None and smoothed_status != previous_smoothed_status:
                # State has changed - reset counters for the previous state
                if previous_smoothed_status == "yawn" and smoothed_status != "yawn":
                    counter_yawn = 0
                    flag_yawn = 0
                    alert_type_yawn = [0,0,0]
                    alert_timer_yawn = datetime.now()
                    last_high_alert_yawn = None
                    print(f"[STATE CHANGE] Resetting yawn detection (changed to {smoothed_status})")
                elif previous_smoothed_status == "eyes_closed" and smoothed_status != "eyes_closed":
                    counter_eye_closed = 0
                    flag_eye_closed = 0
                    alert_type_eye_closed = [0,0,0]
                    alert_timer_eye_closed = datetime.now()
                    last_high_alert_eye_closed = None
                    print(f"[STATE CHANGE] Resetting eyes_closed detection (changed to {smoothed_status})")
                elif previous_smoothed_status == "distracted" and smoothed_status != "distracted":
                    counter_distraction = 0
                    flag_distraction = 0
                    alert_type_distraction = [0,0,0]
                    alert_timer_distraction = datetime.now()
                    last_high_alert_distraction = None
                    print(f"[STATE CHANGE] Resetting distraction detection (changed to {smoothed_status})")
                elif previous_smoothed_status == "phone" and smoothed_status != "phone":
                    counter_phone = 0
                    flag_phone = 0
                    alert_type_phone = [0,0,0]
                    alert_timer_phone = datetime.now()
                    last_high_alert_phone = None
                    print(f"[STATE CHANGE] Resetting phone detection (changed to {smoothed_status})")
                elif previous_smoothed_status == "smoking" and smoothed_status != "smoking":
                    counter_smoking = 0
                    flag_smoking = 0
                    alert_type_smoking = [0,0,0]
                    alert_timer_smoking = datetime.now()
                    last_high_alert_smoking = None
                    print(f"[STATE CHANGE] Resetting smoking detection (changed to {smoothed_status})")
            
            # Update previous state
            previous_smoothed_status = smoothed_status
            
            # with open("log_detection.txt", "r") as file:
            #     while True:
            #         print(counter_awake, counter_yawn, counter_eye_closed, counter_distraction, counter_phone, counter_smoking)
            #         status = file.readline().strip()  # Read one line at a time
            #         if not status:  # Break the loop if end of file is reached
            #             break
            #         print(status)
                    # detection logic
            try:
                if smoothed_status in ("awake", "no_detection"):
                    counter_awake += 1 # no need

                elif smoothed_status == "yawn":
                    # calls functions to update parameters
                    counter_yawn, timer_yawn,flag_yawn, alert_type_yawn, alert_timer_yawn, last_high_alert_yawn = update_status("yawn", counter_yawn, timer_yawn, config_yawn_timer , flag_yawn, alert_timer_yawn, alert_type_yawn, config_low_alert_timer.get("yawn"), config_medium_alert_timer.get("yawn"), config_high_alert_timer.get("yawn"), config_thresholds.get("yawn"), last_high_alert_yawn)

                elif smoothed_status == "eyes_closed":    
                    counter_eye_closed, timer_eye_closed,flag_eye_closed, alert_type_eye_closed, alert_timer_eye_closed, last_high_alert_eye_closed = update_status("eyes closed", counter_eye_closed, timer_eye_closed, config_eyes_closed_timer ,flag_eye_closed, alert_timer_eye_closed, alert_type_eye_closed, config_low_alert_timer.get("eyes_closed"), config_medium_alert_timer.get("eyes_closed"), config_high_alert_timer.get("eyes_closed"), config_thresholds.get("eyes_closed"), last_high_alert_eye_closed)

                elif smoothed_status == "distracted":
                    counter_distraction, timer_distraction,flag_distraction,alert_type_distraction, alert_timer_distraction, last_high_alert_distraction = update_status("distraction", counter_distraction, timer_distraction, config_distraction_timer ,flag_distraction, alert_timer_distraction, alert_type_distraction, config_low_alert_timer.get("distraction"), config_medium_alert_timer.get("distraction"), config_high_alert_timer.get("distraction"), config_thresholds.get("distraction"), last_high_alert_distraction)

                elif smoothed_status == "phone":
                    counter_phone, timer_phone,flag_phone,alert_type_phone,alert_timer_phone, last_high_alert_phone = update_status("phone", counter_phone, timer_phone, config_phone_timer ,flag_phone, alert_timer_phone, alert_type_phone, config_low_alert_timer.get("yawn"), config_medium_alert_timer.get("phone"), config_high_alert_timer.get("phone"), config_thresholds.get("phone"), last_high_alert_phone)

                elif smoothed_status == "smoking":
                    counter_smoking, timer_smoking,flag_smoking, alert_type_smoking,alert_timer_smoking, last_high_alert_smoking = update_status("smoking", counter_smoking, timer_smoking, config_smoking_timer, flag_smoking, alert_timer_smoking, alert_type_smoking, config_low_alert_timer.get("smoking"), config_medium_alert_timer.get("smoking"), config_high_alert_timer.get("smoking"), config_thresholds.get("smoking"), last_high_alert_smoking)
                    
                else:
                    print("No valid text in log_detection.txt")

                #print(counter_yawn) # tested for yawn detection

                time.sleep(config_time_sleep) # easy to read logs in console, but will change the overall logic
            
            except Exception as e:
                print("Error:", e)
                print("Configuration error. Check the configuration file.")
                exit()
        else:
            print("Vehicle speed is less than the configured value. [ Vehicle speed config: ",vehicle_speed_config," Current Vehicle Speed: ", current_vehicle_speed,"]")
            time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target = get_gps_location).start()
    threading.Thread(target = get_gps_location).start()
    run_detection_logic()
