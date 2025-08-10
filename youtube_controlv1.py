import os
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
from collections import deque
import traceback
import platform
import random
import csv

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    selenium_available = True
except ImportError:
    selenium_available = False
    print("Could not import Selenium library. Please install: pip install selenium webdriver-manager")

try:
    from termcolor import colored
    termcolor_available = True
except ImportError:
    termcolor_available = False

# ======== Global Variables ========
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=1)
processing_active = True
fps_values = deque(maxlen=10)
driver = None
video_url = ""
selenium_active = False
browser_type = "brave"
current_speed = 1.0
speed_index = 3
speed_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
current_volume = 1.0
distance_history = deque(maxlen=5)
filtered_distance_history = deque(maxlen=20)
last_speed_change = 0
last_volume_change = 0
MIN_SPEED_CHANGE_INTERVAL = 0.015
MIN_VOLUME_CHANGE_INTERVAL = 0.015
speed_direction_bias = 0
volume_direction_bias = 0
prev_left_hand_distance = None
prev_right_hand_distance = None
last_next_action = 0
last_pause_action = 0
MIN_ACTION_INTERVAL = 2.5
next_gesture_start = None
pause_gesture_start = None
NEXT_GESTURE_DURATION = 1.0
PAUSE_GESTURE_DURATION = 0.7
PAUSE_THRESHOLD_OPEN = 0.15
PAUSE_THRESHOLD_CLOSE = 0.09
NEXT_THRESHOLD = 0.1
THUMB_UP_THRESHOLD = -0.03
VOLUME_CHANGE_THRESHOLD = 0.005
action_status = None
selenium_action_lock = threading.Lock()
log_file = "gesture_log.csv"
log_buffer = []  
last_log_write = 0
LOG_WRITE_INTERVAL = 5.0  
total_frames_processed = 0
frames_with_hands = 0
gesture_counts = {
    "Next": {"success": 0, "total": 0},
    "Pause": {"success": 0, "total": 0},
    "Play": {"success": 0, "total": 0},
    "Speed Up": {"success": 0, "total": 0},
    "Speed Down": {"success": 0, "total": 0},
    "Volume Up": {"success": 0, "total": 0},
    "Volume Down": {"success": 0, "total": 0}
}
frame_processing_times = deque(maxlen=100)

# ======== MediaPipe Hands Setup ========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    static_image_mode=False
)

# ======== Advanced Smooth Filter ========
class AdvancedSmoothFilter:
    def __init__(self, alpha=0.3, responsiveness=0.7, min_alpha=0.1, max_alpha=0.6):
        self.value = None
        self.base_alpha = alpha
        self.responsiveness = responsiveness
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.velocity = 0
        self.acceleration = 0
        self.last_values = deque(maxlen=3)
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
            self.last_values.append(new_value)
            return new_value
        
        old_velocity = self.velocity
        self.velocity = new_value - self.value
        self.acceleration = self.velocity - old_velocity
        
        diff = abs(new_value - self.value)
        direction = 1 if new_value > self.value else -1
        
        adjusted_alpha = max(self.min_alpha, 
                            min(self.max_alpha, 
                                self.base_alpha - diff * self.responsiveness * direction))
        
        filtered_value = adjusted_alpha * new_value + (1 - adjusted_alpha) * self.value
        
        prediction_factor = 0.4
        predicted_value = filtered_value + self.velocity * prediction_factor + self.acceleration * 0.15
        
        max_deviation = 0.12
        if abs(predicted_value - filtered_value) > max_deviation:
            direction = 1 if predicted_value > filtered_value else -1
            predicted_value = filtered_value + (direction * max_deviation)
            
        self.value = filtered_value
        self.last_values.append(filtered_value)
        return predicted_value if self.responsiveness > 0.8 else filtered_value

left_hand_filter = AdvancedSmoothFilter(alpha=0.3, responsiveness=0.7, min_alpha=0.1, max_alpha=0.6)
right_hand_filter = AdvancedSmoothFilter(alpha=0.3, responsiveness=0.7, min_alpha=0.1, max_alpha=0.6)

# ======== Camera Reader ========
def camera_reader():
    global processing_active
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Please check your camera connection.")
            processing_active = False
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while processing_active:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to capture frame from camera. Trying again...")
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass
                
    except Exception as e:
        print(f"ERROR in camera thread: {e}")
        processing_active = False
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        print("Camera thread terminated.")

# ======== Hand Processor ========
def hand_processor():
    global processing_active, current_volume, total_frames_processed, frames_with_hands, frame_processing_times
    while processing_active:
        try:
            frame = frame_queue.get(timeout=0.03)
            start_time = time.time()
            h, w, _ = frame.shape
            
            scale = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            processed_data = {
                'landmarks': [],
                'hand_sides': [],
                'hand_points': [],
                'left_hand_data': None,
                'right_hand_data': None,
                'frame': frame,
                'fps': 0
            }
            
            if results.multi_hand_landmarks and results.multi_handedness:
                frames_with_hands += 1
                for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_side = 'left' if handedness.classification[0].label == "Left" else 'right'
                    processed_data['hand_sides'].append(hand_side)
                    processed_data['landmarks'].append(hand_landmarks)
                    
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    wrist = hand_landmarks.landmark[0]
                    
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    
                    thumb_index_distance = np.hypot(thumb_x - index_x, thumb_y - index_y) / w
                    
                    if hand_side == 'right':
                        processed_data['right_hand_data'] = {
                            'index_point': (index_x, index_y),
                            'thumb_point': (thumb_x, thumb_y),
                            'distance': thumb_index_distance,
                            'volume': current_volume
                        }
                    elif hand_side == 'left':
                        processed_data['left_hand_data'] = {
                            'index_point': (index_x, index_y),
                            'thumb_point': (thumb_x, thumb_y),
                            'wrist_point': (wrist_x, wrist_y),
                            'distance': thumb_index_distance
                        }
            
            total_frames_processed += 1
            elapsed = max(time.time() - start_time, 0.001)
            frame_processing_times.append(elapsed)
            fps_values.append(1.0 / elapsed)
            processed_data['fps'] = int(np.mean(fps_values))
            
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put(processed_data, block=False)
            
        except queue.Empty:
            time.sleep(0.001)
        except Exception as e:
            print(f"Hand processor error: {e}")

# ======== Helper Functions ========
def draw_centered_label(frame, text, position, size=0.5, thickness=1):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0]
    text_x, text_y = position
    
    bg_width = text_size[0] + 10
    bg_height = text_size[1] + 10
    bg_x = text_x - bg_width // 2
    bg_y = text_y - bg_height // 2
    
    cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height), (255, 255, 255), -1)
    
    text_offset_x = bg_x + (bg_width - text_size[0]) // 2
    text_offset_y = bg_y + (bg_height + text_size[1]) // 2
    cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness)

def get_browser_user_data_dir(browser_type="brave"):
    system = platform.system()
    if browser_type.lower() == "brave":
        if system == "Windows":
            return os.path.join(os.environ["LOCALAPPDATA"], "BraveSoftware", "Brave-Browser", "User Data")
        elif system == "Darwin":
            return os.path.expanduser("~/Library/Application Support/BraveSoftware/Brave-Browser")
        elif system == "Linux":
            return os.path.expanduser("~/.config/BraveSoftware/Brave-Browser")
    else:
        if system == "Windows":
            return os.path.join(os.environ["LOCALAPPDATA"], "Google", "Chrome", "User Data")
        elif system == "Darwin":
            return os.path.expanduser("~/Library/Application Support/Google/Chrome")
        elif system == "Linux":
            return os.path.expanduser("~/.config/google-chrome")
    return None

def setup_selenium():
    global driver, selenium_active, browser_type, video_url, current_volume
    if not selenium_available:
        print("Selenium is not available - skipping browser initialization")
        return False
    
    print("\n=== MỞ VIDEO TRÊN BRAVE ===")
    video_url = input("\nEnter YouTube video URL (press Enter to use default video): ").strip()
    if not video_url:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"Using default video: {video_url}")
    
    temp_user_dir = os.path.join(os.getcwd(), "temp_selenium_profile")
    os.makedirs(temp_user_dir, exist_ok=True)
    
    options = webdriver.ChromeOptions()
    if browser_type == "brave":
        brave_paths = [
            "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            "C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            os.path.join(os.environ["LOCALAPPDATA"], "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
        ]
        for path in brave_paths:
            if os.path.exists(path):
                options.binary_location = path
                break
    
    options.add_argument(f"--user-data-dir={temp_user_dir}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    try:
        print("\n🔄 Starting browser... Please wait...")
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
        except Exception as e:
            print(f"Lỗi khi sử dụng ChromeDriverManager: {e}")
            driver = webdriver.Chrome(options=options)
        
        print(f"\n🌐 Opening YouTube video: {video_url}")
        driver.get(video_url)
        
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            print("✅ Video loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Video element not found - {str(e)}")
        
        if inject_controller_script():
            selenium_active = True
            try:
                current_volume = driver.execute_script("return document.querySelector('video') ? document.querySelector('video').volume : 1.0;")
                print(f"Current volume: {current_volume:.2f}")
            except Exception as e:
                print(f"⚠️ Error getting initial volume: {e}")
                current_volume = 1.0
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo Selenium: {e}")
        traceback.print_exc()
        return False

def inject_controller_script():
    global driver, current_speed, current_volume
    if not driver:
        return False
    try:
        driver.title
        time.sleep(random.uniform(0.5, 2.0))
        action = webdriver.ActionChains(driver)
        action.move_by_offset(
            random.randint(10, 50),
            random.randint(10, 50)
        ).perform()
        
        controller_script = """
        if (!document.getElementById('ai-speed-controller')) {
            window.aiHandController = {
                currentSpeed: document.querySelector('video') ? document.querySelector('video').playbackRate : 1.0,
                currentVolume: document.querySelector('video') ? document.querySelector('video').volume : 1.0,
                updateQueue: [],
                volumeUpdateQueue: [],
                processingUpdate: false,
                processingVolumeUpdate: false,
                lastUpdateTime: Date.now(),
                lastVolumeUpdateTime: Date.now(),
                pendingAnimationFrame: null,
                pendingVolumeAnimationFrame: null
            };
            
            const controlPanel = document.createElement('div');
            controlPanel.id = 'ai-speed-controller';
            controlPanel.style.position = 'fixed';
            controlPanel.style.bottom = '80px';
            controlPanel.style.right = '20px';
            controlPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
            controlPanel.style.color = 'white';
            controlPanel.style.padding = '15px';
            controlPanel.style.borderRadius = '10px';
            controlPanel.style.zIndex = '9999';
            controlPanel.style.display = 'flex';
            controlPanel.style.flexDirection = 'column';
            controlPanel.style.alignItems = 'center';
            controlPanel.style.fontFamily = 'Arial, sans-serif';
            controlPanel.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
            controlPanel.style.transition = 'background-color 0.15s';
            
            const title = document.createElement('div');
            title.textContent = 'AI Hand Controller';
            title.style.fontWeight = 'bold';
            title.style.fontSize = '14px';
            title.style.marginBottom = '10px';
            controlPanel.appendChild(title);
            
            const speedDisplay = document.createElement('div');
            speedDisplay.id = 'current-speed-display';
            speedDisplay.textContent = `Speed: ${window.aiHandController.currentSpeed.toFixed(2)}x`;
            speedDisplay.style.margin = '5px 0';
            speedDisplay.style.fontSize = '16px';
            controlPanel.appendChild(speedDisplay);
            
            const volumeDisplay = document.createElement('div');
            volumeDisplay.id = 'current-volume-display';
            volumeDisplay.textContent = `Volume: ${(window.aiHandController.currentVolume * 100).toFixed(0)}%`;
            volumeDisplay.style.margin = '5px 0';
            controlPanel.style.fontSize = '16px';
            controlPanel.appendChild(volumeDisplay);
            
            const buttonContainer = document.createElement('div');
            buttonContainer.style.display = 'flex';
            buttonContainer.style.justifyContent = 'center';
            buttonContainer.style.width = '100%';
            buttonContainer.style.marginTop = '5px';
            
            const decreaseSpeedBtn = document.createElement('button');
            decreaseSpeedBtn.textContent = '-';
            decreaseSpeedBtn.style.margin = '0 5px';
            decreaseSpeedBtn.style.padding = '8px 15px';
            decreaseSpeedBtn.style.backgroundColor = '#c00';
            decreaseSpeedBtn.style.color = 'white';
            decreaseSpeedBtn.style.border = 'none';
            decreaseSpeedBtn.style.borderRadius = '5px';
            decreaseSpeedBtn.style.cursor = 'pointer';
            decreaseSpeedBtn.style.fontSize = '16px';
            decreaseSpeedBtn.style.fontWeight = 'bold';
            
            const increaseSpeedBtn = document.createElement('button');
            increaseSpeedBtn.textContent = '+';
            increaseSpeedBtn.style.margin = '0 5px';
            increaseSpeedBtn.style.padding = '8px 15px';
            increaseSpeedBtn.style.backgroundColor = '#c00';
            increaseSpeedBtn.style.color = 'white';
            increaseSpeedBtn.style.border = 'none';
            increaseSpeedBtn.style.borderRadius = '5px';
            increaseSpeedBtn.style.cursor = 'pointer';
            increaseSpeedBtn.style.fontSize = '16px';
            increaseSpeedBtn.style.fontWeight = 'bold';
            
            const decreaseVolumeBtn = document.createElement('button');
            decreaseVolumeBtn.textContent = '🔉';
            decreaseVolumeBtn.style.margin = '0 5px';
            decreaseVolumeBtn.style.padding = '8px 15px';
            decreaseVolumeBtn.style.backgroundColor = '#c00';
            decreaseVolumeBtn.style.color = 'white';
            decreaseVolumeBtn.style.border = 'none';
            decreaseVolumeBtn.style.borderRadius = '5px';
            decreaseVolumeBtn.style.cursor = 'pointer';
            decreaseVolumeBtn.style.fontSize = '16px';
            decreaseVolumeBtn.style.fontWeight = 'bold';
            
            const increaseVolumeBtn = document.createElement('button');
            increaseVolumeBtn.textContent = '🔊';
            increaseVolumeBtn.style.margin = '0 5px';
            increaseVolumeBtn.style.padding = '8px 15px';
            increaseVolumeBtn.style.backgroundColor = '#c00';
            increaseVolumeBtn.style.color = 'white';
            increaseVolumeBtn.style.border = 'none';
            increaseVolumeBtn.style.borderRadius = '5px';
            increaseVolumeBtn.style.cursor = 'pointer';
            increaseVolumeBtn.style.fontSize = '16px';
            increaseVolumeBtn.style.fontWeight = 'bold';
            
            buttonContainer.appendChild(decreaseSpeedBtn);
            buttonContainer.appendChild(increaseSpeedBtn);
            buttonContainer.appendChild(decreaseVolumeBtn);
            buttonContainer.appendChild(increaseVolumeBtn);
            controlPanel.appendChild(buttonContainer);
            
            document.body.appendChild(controlPanel);
            
            function processUpdateQueue() {
                if (window.aiHandController.processingUpdate) return;
                
                if (window.aiHandController.updateQueue.length > 0) {
                    window.aiHandController.processingUpdate = true;
                    
                    const latestRate = window.aiHandController.updateQueue.pop();
                    window.aiHandController.updateQueue = [];
                    
                    const video = document.querySelector('video');
                    if (video) {
                        video.playbackRate = latestRate;
                        window.aiHandController.currentSpeed = latestRate;
                        
                        if (window.aiHandController.pendingAnimationFrame === null) {
                            window.aiHandController.pendingAnimationFrame = requestAnimationFrame(() => {
                                const display = document.getElementById('current-speed-display');
                                if (display) {
                                    display.textContent = `Speed: ${latestRate.toFixed(2)}x`;
                                }
                                controlPanel.style.backgroundColor = 'rgba(204, 0, 0, 0.9)';
                                setTimeout(() => {
                                    controlPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                                }, 150);
                                window.aiHandController.pendingAnimationFrame = null;
                            });
                        }
                    }
                    
                    window.aiHandController.processingUpdate = false;
                }
            }
            
            function processVolumeUpdateQueue() {
                if (window.aiHandController.processingVolumeUpdate) return;
                
                if (window.aiHandController.volumeUpdateQueue.length > 0) {
                    window.aiHandController.processingVolumeUpdate = true;
                    
                    const latestVolume = window.aiHandController.volumeUpdateQueue.pop();
                    window.aiHandController.volumeUpdateQueue = [];
                    
                    const video = document.querySelector('video');
                    if (video) {
                        video.volume = latestVolume;
                        window.aiHandController.currentVolume = latestVolume;
                        
                        if (window.aiHandController.pendingVolumeAnimationFrame === null) {
                            window.aiHandController.pendingVolumeAnimationFrame = requestAnimationFrame(() => {
                                const display = document.getElementById('current-volume-display');
                                if (display) {
                                    display.textContent = `Volume: ${(latestVolume * 100).toFixed(0)}%`;
                                }
                                controlPanel.style.backgroundColor = 'rgba(204, 0, 0, 0.9)';
                                setTimeout(() => {
                                    controlPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                                }, 150);
                                window.aiHandController.pendingVolumeAnimationFrame = null;
                            });
                        }
                    }
                    
                    window.aiHandController.processingVolumeUpdate = false;
                }
            }
            
            window.updatePlaybackSpeed = function(rate) {
                if (typeof rate !== 'number' || isNaN(rate) || rate < 0.25 || rate > 2.0) {
                    return false;
                }
                window.aiHandController.updateQueue.push(rate);
                processUpdateQueue();
                return true;
            };
            
            window.updateVolume = function(volume) {
                if (typeof volume !== 'number' || isNaN(volume) || volume < 0.0 || volume > 1.0) {
                    return false;
                }
                window.aiHandController.volumeUpdateQueue.push(volume);
                processVolumeUpdateQueue();
                return true;
            };
            
            const video = document.querySelector('video');
            if (video) {
                video.addEventListener('ratechange', function() {
                    if (Math.abs(video.playbackRate - window.aiHandController.currentSpeed) > 0.01) {
                        window.aiHandController.currentSpeed = video.playbackRate;
                        const display = document.getElementById('current-speed-display');
                        if (display) {
                            display.textContent = `Speed: ${video.playbackRate.toFixed(2)}x`;
                        }
                    }
                });
                video.addEventListener('volumechange', function() {
                    if (Math.abs(video.volume - window.aiHandController.currentVolume) > 0.01) {
                        window.aiHandController.currentVolume = video.volume;
                        const display = document.getElementById('current-volume-display');
                        if (display) {
                            display.textContent = `Volume: ${(video.volume * 100).toFixed(0)}%`;
                        }
                    }
                });
            }
            
            decreaseSpeedBtn.addEventListener('click', function() {
                const video = document.querySelector('video');
                if (video && video.playbackRate > 0.25) {
                    window.updatePlaybackSpeed(Math.max(0.25, video.playbackRate - 0.25));
                }
            });
            
            increaseSpeedBtn.addEventListener('click', function() {
                const video = document.querySelector('video');
                if (video && video.playbackRate < 2.0) {
                    window.updatePlaybackSpeed(Math.min(2.0, video.playbackRate + 0.25));
                }
            });
            
            decreaseVolumeBtn.addEventListener('click', function() {
                const video = document.querySelector('video');
                if (video && video.volume > 0.0) {
                    window.updateVolume(Math.max(0.0, video.volume - 0.1));
                }
            });
            
            increaseVolumeBtn.addEventListener('click', function() {
                const video = document.querySelector('video');
                if (video && video.volume < 1.0) {
                    window.updateVolume(Math.min(1.0, video.volume + 0.1));
                }
            });
            
            window.setYouTubeSpeed = function(speed) {
                try {
                    if (typeof speed !== 'number' || isNaN(speed) || speed < 0.25 || speed > 2.0) {
                        return false;
                    }
                    if (Math.abs(speed - window.aiHandController.currentSpeed) > 0.01) {
                        window.updatePlaybackSpeed(speed);
                    }
                    return true;
                } catch (e) {
                    console.error('Error in setYouTubeSpeed:', e);
                    return false;
                }
            };
            
            window.setYouTubeVolume = function(volume) {
                try {
                    if (typeof volume !== 'number' || isNaN(volume) || volume < 0.0 || volume > 1.0) {
                        return false;
                    }
                    if (Math.abs(volume - window.aiHandController.currentVolume) > 0.01) {
                        window.updateVolume(volume);
                    }
                    return true;
                } catch (e) {
                    console.error('Error in setYouTubeVolume:', e);
                    return false;
                }
            };
            
            document.addEventListener('keydown', function(e) {
                if (e.key === '.' || e.key === '>') {
                    const video = document.querySelector('video');
                    if (video && video.playbackRate < 2.0) {
                        window.updatePlaybackSpeed(Math.min(2.0, video.playbackRate + 0.25));
                    }
                } else if (e.key === ',' || e.key === '<') {
                    const video = document.querySelector('video');
                    if (video && video.playbackRate > 0.25) {
                        window.updatePlaybackSpeed(Math.max(0.25, video.playbackRate - 0.25));
                    }
                } else if (e.key === '-') {
                    const video = document.querySelector('video');
                    if (video && video.volume > 0.0) {
                        window.updateVolume(Math.max(0.0, video.volume - 0.1));
                    }
                } else if (e.key === '=') {
                    const video = document.querySelector('video');
                    if (video && video.volume < 1.0) {
                        window.updateVolume(Math.min(1.0, video.volume + 0.1));
                    }
                }
            });
            
            console.log('AI Hand Controller added to YouTube!');
            return true;
        }
        return true;
        """
        
        print("👉 Injecting controller script...")
        driver.execute_script(controller_script)
        print("Added speed and volume control panel to YouTube!")
        
        try:
            current_speed = driver.execute_script("return document.querySelector('video') ? document.querySelector('video').playbackRate : 1.0;")
            current_volume = driver.execute_script("return document.querySelector('video') ? document.querySelector('video').volume : 1.0;")
            print(f"Current playback speed: {current_speed}x")
            print(f"Current volume: {current_volume:.2f}")
        except Exception as e:
            print(f"⚠️ Error getting initial playback speed or volume: {e}")
            current_speed = 1.0
            current_volume = 1.0
        
        return True
    except Exception as e:
        print(f"Error adding control panel: {e}")
        return False

def change_youtube_speed(new_speed):
    global driver, selenium_active, current_speed
    if not driver or not selenium_active:
        print("Selenium not active or driver not initialized")
        return False
    try:
        driver.title
        if not isinstance(new_speed, (int, float)) or new_speed < 0.25 or new_speed > 2.0:
            print(f"Invalid speed value: {new_speed}")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = driver.execute_script(f"return window.setYouTubeSpeed({new_speed});")
                if success:
                    current_speed = new_speed
                    return True
                else:
                    print(f"Attempt {attempt + 1}: setYouTubeSpeed returned false")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error changing speed: {e}")
                time.sleep(0.5)
        
        try:
            driver.title
        except:
            print("Selenium driver crashed")
            selenium_active = False
        return False
    except Exception as e:
        print(f"❌ Lỗi khi điều chỉnh tốc độ video: {e}")
        try:
            driver.title
        except:
            print("Selenium driver crashed")
            selenium_active = False
        return False

def change_youtube_volume(new_volume):
    global driver, selenium_active, current_volume
    if not driver or not selenium_active:
        print("Selenium not active or driver not initialized")
        return False
    try:
        driver.title
        if not isinstance(new_volume, (int, float)) or new_volume < 0.0 or new_volume > 1.0:
            print(f"Invalid volume value: {new_volume}")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = driver.execute_script(f"return window.setYouTubeVolume({new_volume});")
                if success:
                    current_volume = new_volume
                    return True
                else:
                    print(f"Attempt {attempt + 1}: setYouTubeVolume returned false")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error changing volume: {e}")
                time.sleep(0.5)
        
        try:
            driver.title
        except:
            print("Selenium driver crashed")
            selenium_active = False
        return False
    except Exception as e:
        print(f"❌ Lỗi khi điều chỉnh âm lượng video: {e}")
        try:
            driver.title
        except:
            print("Selenium driver crashed")
            selenium_active = False
        return False

def perform_next_action():
    global driver, selenium_active, action_status, last_next_action, gesture_counts
    
    current_time = time.time()
    if current_time - last_next_action < MIN_ACTION_INTERVAL:
        return False  

    last_next_action = time.time()
    start_time = time.time()

    if not driver or not selenium_active:
        print("Selenium not active or driver not initialized")
        log_gesture_result("Next", False, 0, int(np.mean(fps_values)), "Selenium not active", 0)
        gesture_counts["Next"]["total"] += 1
        return False

    with selenium_action_lock:
        try:
            print("INFO: Chuyen video bang phim tat...")
            
            actions = ActionChains(driver)
            actions.key_down(Keys.SHIFT).send_keys('n').key_up(Keys.SHIFT).perform()
            
            action_status = "Next Video"
            print("✅ Da chuyen video bang phim tat.")
            
            latency = time.time() - start_time
            log_gesture_result("Next", True, latency, int(np.mean(fps_values)), action_status, latency)
            gesture_counts["Next"]["success"] += 1
            gesture_counts["Next"]["total"] += 1
            
            return True
            
        except Exception as e:
            # Nếu có lỗi xảy ra với phương pháp phím tắt
            print(f"❌ Loi khi dung phim tat: {e}")
            action_status = "Next video failed"
            log_gesture_result("Next", False, 0, int(np.mean(fps_values)), f"Error: {str(e)}", 0)
            gesture_counts["Next"]["total"] += 1
            return False

# THAY THẾ TOÀN BỘ HÀM NÀY

def perform_pause_action():
    global driver, selenium_active, action_status, last_pause_action, gesture_counts
    
    current_time = time.time()
    if current_time - last_pause_action < MIN_ACTION_INTERVAL:
        return False # Vẫn trong thời gian chờ, không làm gì cả

    # Cập nhật thời gian ngay khi một hành động được "thử"
    last_pause_action = time.time()
    start_time = time.time()

    if not driver or not selenium_active:
        print("Selenium not active or driver not initialized")
        return False

    with selenium_action_lock:
        try:
            # Lấy trạng thái video TRƯỚC KHI thực hiện hành động
            is_paused_before_action = driver.execute_script("return document.querySelector('video').paused;")
            gesture_name = "Play" if is_paused_before_action else "Pause"
            
            print(f"INFO: Thuc hien hanh dong '{gesture_name}' bang phim tat (K)...")
            
            # Gửi phím 'k' đến trang web
            actions = ActionChains(driver)
            actions.send_keys('k').perform()

            # Cập nhật trạng thái và ghi log
            action_status = "Paused" if not is_paused_before_action else "Playing"
            print(f"✅ Video da duoc {'tam dung' if not is_paused_before_action else 'phat'}.")
            
            latency = time.time() - start_time
            log_gesture_result(gesture_name, True, latency, int(np.mean(fps_values)), action_status, latency)
            gesture_counts[gesture_name]["success"] += 1
            gesture_counts[gesture_name]["total"] += 1
            return True
            
        except Exception as e:
            print(f"❌ Loi khi dung phim tat Pause/Play: {e}")
            # Cố gắng xác định gesture_name lần nữa để ghi log lỗi chính xác
            try:
                is_paused = driver.execute_script("return document.querySelector('video').paused;")
                gesture_name_on_error = "Play" if is_paused else "Pause"
            except:
                gesture_name_on_error = "Pause/Play Action" # Tên chung nếu không kiểm tra được

            log_gesture_result(gesture_name_on_error, False, 0, int(np.mean(fps_values)), f"Error: {str(e)}", 0)
            if gesture_name_on_error in gesture_counts:
                gesture_counts[gesture_name_on_error]["total"] += 1
            return False
def log_gesture_result(gesture, success, latency, fps, action_status, selenium_latency):
    global log_buffer, last_log_write, total_frames_processed, frames_with_hands, filtered_distance_history, frame_processing_times
    hand_detection_accuracy = frames_with_hands / max(total_frames_processed, 1)
    frame_processing_rate = total_frames_processed / max(total_frames_processed, 1)
    distance_stability = np.std(list(filtered_distance_history)) if filtered_distance_history else 0
    avg_frame_processing_time = np.mean(list(frame_processing_times)) if frame_processing_times else 0
    gesture_success_rate = gesture_counts[gesture]["success"] / max(gesture_counts[gesture]["total"], 1) if gesture_counts[gesture]["total"] > 0 else 0
    log_buffer.append([
        time.strftime('%Y-%m-%d %H:%M:%S'),
        gesture,
        success,
        f"{latency:.3f}",
        fps,
        action_status,
        f"{selenium_latency:.3f}",
        f"{hand_detection_accuracy:.3f}",
        f"{frame_processing_rate:.3f}",
        f"{distance_stability:.6f}",
        f"{avg_frame_processing_time:.6f}",
        f"{gesture_success_rate:.3f}"
    ])
    current_time = time.time()
    if current_time - last_log_write >= LOG_WRITE_INTERVAL or len(log_buffer) >= 100:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for log_entry in log_buffer:
                writer.writerow(log_entry)
        log_buffer.clear()
        last_log_write = current_time

def async_action(action_func, *args):
    """Run a Selenium action in a separate thread to avoid blocking the main loop."""
    def wrapper():
        action_func(*args)
    threading.Thread(target=wrapper, daemon=True).start()

def adjust_playback_speed(direction, distance_change=None):
    global speed_index, current_speed, speed_direction_bias, speed_values, gesture_counts
    start_time = time.time()
    if distance_change is not None:
        change_magnitude = abs(distance_change) * 100
        change_probability = min(1.0, change_magnitude ** 1.4 / 35)
        
        if distance_change > 0 and np.random.random() < change_probability and speed_index < len(speed_values) - 1:
            speed_index += 1
            current_speed = speed_values[speed_index]
            if selenium_active:
                async_action(change_youtube_speed, current_speed)
            latency = time.time() - start_time
            log_gesture_result("Speed Up", True, latency, int(np.mean(fps_values)), f"Speed: {current_speed}x", 0)
            gesture_counts["Speed Up"]["success"] += 1
            gesture_counts["Speed Up"]["total"] += 1
            return current_speed
        elif distance_change < 0 and np.random.random() < change_probability and speed_index > 0:
            speed_index -= 1
            current_speed = speed_values[speed_index]
            if selenium_active:
                async_action(change_youtube_speed, current_speed)
            latency = time.time() - start_time
            log_gesture_result("Speed Down", True, latency, int(np.mean(fps_values)), f"Speed: {current_speed}x", 0)
            gesture_counts["Speed Down"]["success"] += 1
            gesture_counts["Speed Down"]["total"] += 1
            return current_speed
        else:
            log_gesture_result("Speed Up" if distance_change > 0 else "Speed Down", False, 0, int(np.mean(fps_values)), "No speed change", 0)
            gesture_counts["Speed Up" if distance_change > 0 else "Speed Down"]["total"] += 1
    
    speed_direction_bias += 1.8 if direction == "faster" else -1.8
    speed_direction_bias = max(-4.0, min(4.0, speed_direction_bias))
    
    should_change = False
    if direction == "faster" and speed_index < len(speed_values) - 1:
        if speed_direction_bias >= 1.2:
            speed_index += 1
            speed_direction_bias = 0
            should_change = True
    elif direction == "slower" and speed_index > 0:
        if speed_direction_bias <= -1.2:
            speed_index -= 1
            speed_direction_bias = 0
            should_change = True
    
    if should_change:
        current_speed = speed_values[speed_index]
        if selenium_active:
            async_action(change_youtube_speed, current_speed)
        latency = time.time() - start_time
        log_gesture_result("Speed Up" if direction == "faster" else "Speed Down", True, latency, int(np.mean(fps_values)), f"Speed: {current_speed}x", 0)
        gesture_counts["Speed Up" if direction == "faster" else "Speed Down"]["success"] += 1
        gesture_counts["Speed Up" if direction == "faster" else "Speed Down"]["total"] += 1
    
    return current_speed

def adjust_volume(direction, distance_change=None):
    global current_volume, volume_direction_bias, gesture_counts
    start_time = time.time()
    if distance_change is not None:
        change_magnitude = abs(distance_change) * 100
        change_probability = min(1.0, change_magnitude ** 1.4 / 35)
        
        if distance_change > 0 and np.random.random() < change_probability and current_volume < 1.0:
            new_volume = min(1.0, current_volume + 0.1)
            if selenium_active:
                async_action(change_youtube_volume, new_volume)
            latency = time.time() - start_time
            log_gesture_result("Volume Up", True, latency, int(np.mean(fps_values)), f"Volume: {int(new_volume * 100)}%", 0)
            gesture_counts["Volume Up"]["success"] += 1
            gesture_counts["Volume Up"]["total"] += 1
            return new_volume
        elif distance_change < 0 and np.random.random() < change_probability and current_volume > 0.0:
            new_volume = max(0.0, current_volume - 0.1)
            if selenium_active:
                async_action(change_youtube_volume, new_volume)
            latency = time.time() - start_time
            log_gesture_result("Volume Down", True, latency, int(np.mean(fps_values)), f"Volume: {int(new_volume * 100)}%", 0)
            gesture_counts["Volume Down"]["success"] += 1
            gesture_counts["Volume Down"]["total"] += 1
            return new_volume
        else:
            log_gesture_result("Volume Up" if distance_change > 0 else "Volume Down", False, 0, int(np.mean(fps_values)), "No volume change", 0)
            gesture_counts["Volume Up" if distance_change > 0 else "Volume Down"]["total"] += 1
    
    volume_direction_bias += 1.8 if direction == "louder" else -1.8
    volume_direction_bias = max(-4.0, min(4.0, volume_direction_bias))
    
    should_change = False
    if direction == "louder" and current_volume < 1.0:
        if volume_direction_bias >= 1.2:
            new_volume = min(1.0, current_volume + 0.1)
            volume_direction_bias = 0
            should_change = True
    elif direction == "quieter" and current_volume > 0.0:
        if volume_direction_bias <= -1.2:
            new_volume = max(0.0, current_volume - 0.1)
            volume_direction_bias = 0
            should_change = True
    
    if should_change:
        current_volume = new_volume
        if selenium_active:
            async_action(change_youtube_volume, new_volume)
        latency = time.time() - start_time
        log_gesture_result("Volume Up" if direction == "louder" else "Volume Down", True, latency, int(np.mean(fps_values)), f"Volume: {int(new_volume * 100)}%", 0)
        gesture_counts["Volume Up" if direction == "louder" else "Volume Down"]["success"] += 1
        gesture_counts["Volume Up" if direction == "louder" else "Volume Down"]["total"] += 1
    
    return current_volume

# ======== Main ========
def main():
    global processing_active, selenium_active, current_speed, current_volume, prev_left_hand_distance, prev_right_hand_distance
    global last_speed_change, speed_direction_bias, last_volume_change, volume_direction_bias, action_status
    global next_gesture_start, pause_gesture_start, log_buffer, last_log_write, total_frames_processed, frames_with_hands
    
    last_speed_status = ""
    last_volume_status = ""
    speed_trend = 0
    volume_trend = 0
    
    # Initialize log file
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gesture", "Success", "Latency", "FPS", "Action Status", "Selenium Latency", 
                         "Hand Detection Accuracy", "Frame Processing Rate", "Distance Stability", "Frame Processing Time", 
                         "Gesture Success Rate"])
    
    if not setup_selenium():
        print("⚠️ Trình duyệt không kết nối được hoặc không mở video.")
        return
    
    print("\n===== USER INTENT =====")
    print("YouTube playback speed and pause/play control (left hand):")
    print("- Increase speed: Move thumb and index finger apart")
    print("- Decrease speed: Pinch thumb and index finger closer (but not too close)")
    print("- Pause: Pinch thumb and index finger close together for 0.7s")
    print("- Play: Spread thumb and index finger apart for 0.7s")
    print("- Next video: Raise both hands for 1.0s")
    print("Volume control (right hand):")
    print("- Increase volume: Move thumb and index finger apart")
    print("- Decrease volume: Pinch thumb and index finger closer")
    print("\nSystem ready! Press ESC to exit.")
    
    camera_thread = threading.Thread(target=camera_reader, daemon=True)
    processor_thread = threading.Thread(target=hand_processor, daemon=True)
    camera_thread.start()
    processor_thread.start()
    
    cv2.namedWindow("Hand Controller", cv2.WINDOW_NORMAL)
    
    try:
        while processing_active:
            try:
                result = result_queue.get(timeout=0.03)
                frame = result.get('frame')
                fps = result.get('fps', 0)
                landmarks = result.get('landmarks')
                hand_sides = result.get('hand_sides')
                left_hand_data = result.get('left_hand_data')
                right_hand_data = result.get('right_hand_data')
                
                if frame is None:
                    continue
                
                h, w, _ = frame.shape
                
                # Draw landmarks and labels
                for hand_landmarks, hand_side in zip(landmarks, hand_sides):
                    color = (0, 255, 0) if hand_side == 'left' else (0, 0, 255)
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=5),
                        mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
                    
                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    draw_centered_label(frame, f"{hand_side.capitalize()} hand", 
                                      (wrist_x, wrist_y - 15), size=0.5, thickness=1)
                
                # Process playback speed, pause/play, and next video (left hand)
                if left_hand_data:
                    index_point = left_hand_data['index_point']
                    thumb_point = left_hand_data['thumb_point']
                    distance = left_hand_data['distance']
                    
                    smoothed_distance = left_hand_filter.update(distance)
                    filtered_distance_history.append(smoothed_distance)
                    
                    # Draw connection between index and thumb for left hand
                    cv2.line(frame, index_point, thumb_point, (0, 255, 255), 2)
                    cv2.circle(frame, index_point, 5, (0, 255, 255), -1)
                    cv2.circle(frame, thumb_point, 5, (0, 255, 255), -1)
                    
                    # Display speed at midpoint
                    mid_x = (index_point[0] + thumb_point[0]) // 2
                    mid_y = (index_point[1] + thumb_point[1]) // 2
                    draw_centered_label(frame, f"{current_speed}x", (mid_x, mid_y), size=0.6, thickness=2)
                    
                    # Speed bar
                    speed_bar_x = 50
                    speed_bar_y = (h - 200) // 2
                    speed_bar_h = 200
                    speed_bar_w = 30
                    
                    cv2.rectangle(frame, (speed_bar_x, speed_bar_y), 
                                (speed_bar_x + speed_bar_w, speed_bar_y + speed_bar_h), 
                                (200, 200, 200), -1)
                    
                    normalized_speed = (speed_index / (len(speed_values) - 1))
                    fill_h = int(speed_bar_h * normalized_speed)
                    cv2.rectangle(frame, (speed_bar_x, speed_bar_y + speed_bar_h - fill_h), 
                                (speed_bar_x + speed_bar_w, speed_bar_y + speed_bar_h),
                                (255, 165, 0), -1)
                    
                    draw_centered_label(frame, f"{current_speed}x", 
                                      (speed_bar_x + speed_bar_w // 2, speed_bar_y + speed_bar_h + 15), 0.5, 1)
                    
                    # Next video detection (both hands raised)
                    current_time = time.time()
                    
                    if left_hand_data and right_hand_data:
                        print(f"Next gesture detected: Both hands raised")
                        if next_gesture_start is None:
                            next_gesture_start = current_time
                        elif current_time - next_gesture_start >= NEXT_GESTURE_DURATION and \
                             current_time - last_next_action >= MIN_ACTION_INTERVAL:
                            async_action(perform_next_action)
                            next_gesture_start = None
                        if next_gesture_start is not None:
                            remaining = NEXT_GESTURE_DURATION - (current_time - next_gesture_start)
                            if remaining > 0:
                                draw_centered_label(frame, f"Next: {remaining:.1f}s", (w // 2, 80), 0.6, 2)
                    elif next_gesture_start is not None:
                        log_gesture_result("Next", False, 0, fps, "No both hands detected", 0)
                        gesture_counts["Next"]["total"] += 1
                        next_gesture_start = None
                    
                    # Pause/Play detection
                    if smoothed_distance < PAUSE_THRESHOLD_CLOSE:
                        if pause_gesture_start is None:
                            pause_gesture_start = current_time
                        elif current_time - pause_gesture_start >= PAUSE_GESTURE_DURATION and \
                             current_time - last_pause_action >= MIN_ACTION_INTERVAL:
                            if selenium_active:
                                video = WebDriverWait(driver, 5).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                                )
                                is_paused = driver.execute_script("return arguments[0].paused;", video)
                                if not is_paused:
                                    async_action(perform_pause_action)
                                    pause_gesture_start = None
                        if pause_gesture_start is not None:
                            remaining = PAUSE_GESTURE_DURATION - (current_time - pause_gesture_start)
                            if remaining > 0:
                                draw_centered_label(frame, f"Pause: {remaining:.1f}s", (w // 2, 80), 0.6, 2)
                    elif smoothed_distance > PAUSE_THRESHOLD_OPEN:
                        if pause_gesture_start is None:
                            pause_gesture_start = current_time
                        elif current_time - pause_gesture_start >= PAUSE_GESTURE_DURATION and \
                             current_time - last_pause_action >= MIN_ACTION_INTERVAL:
                            if selenium_active:
                                video = WebDriverWait(driver, 5).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                                )
                                is_paused = driver.execute_script("return arguments[0].paused;", video)
                                if is_paused:
                                    async_action(perform_pause_action)
                                    pause_gesture_start = None
                        if pause_gesture_start is not None:
                            remaining = PAUSE_GESTURE_DURATION - (current_time - pause_gesture_start)
                            if remaining > 0:
                                draw_centered_label(frame, f"Play: {remaining:.1f}s", (w // 2, 80), 0.6, 2)
                    elif pause_gesture_start is not None:
                        log_gesture_result("Pause" if smoothed_distance < PAUSE_THRESHOLD_CLOSE else "Play", False, 0, fps, "Distance not in threshold", 0)
                        gesture_counts["Pause" if smoothed_distance < PAUSE_THRESHOLD_CLOSE else "Play"]["total"] += 1
                        pause_gesture_start = None
                    
                    # Process speed control
                    if prev_left_hand_distance is not None:
                        distance_change = smoothed_distance - prev_left_hand_distance
                        
                        if abs(distance_change) > 0.005:
                            speed_trend = 1 if distance_change > 0 else -1
                        else:
                            speed_trend = 0
                        
                        dynamic_threshold = 0.0025 + 0.002 * (1 - abs(distance_change) * 12)
                        dynamic_threshold = max(0.002, min(0.005, dynamic_threshold))
                        
                        if current_time - last_speed_change > MIN_SPEED_CHANGE_INTERVAL:
                            if abs(distance_change) > dynamic_threshold and \
                               abs(smoothed_distance - PAUSE_THRESHOLD_CLOSE) > 0.02:
                                direction = "faster" if distance_change > 0 else "slower"
                                old_speed = current_speed
                                current_speed = adjust_playback_speed(direction, distance_change)
                                
                                if current_speed != old_speed:
                                    last_speed_status = "Speed up" if current_speed > old_speed else "Slow down"
                                last_speed_change = current_time
                        
                        if speed_trend > 0:
                            draw_centered_label(frame, "▲", 
                                             (speed_bar_x + speed_bar_w // 2, speed_bar_y - 15), 0.7, 2)
                        elif speed_trend < 0:
                            draw_centered_label(frame, "▼", 
                                             (speed_bar_x + speed_bar_w // 2, speed_bar_y - 15), 0.7, 2)
                    
                    prev_left_hand_distance = smoothed_distance
                    
                    if last_speed_status:
                        color = (255, 165, 0) if last_speed_status == "Speed up" else (0, 165, 255)
                        draw_centered_label(frame, last_speed_status, 
                                         (speed_bar_x + speed_bar_w // 2, speed_bar_y - 35), 0.5, 1)
                
                # Volume control (right hand)
                if right_hand_data:
                    index_point = right_hand_data['index_point']
                    thumb_point = right_hand_data['thumb_point']
                    distance = right_hand_data['distance']
                    current_volume = right_hand_data['volume']
                    
                    smoothed_distance = right_hand_filter.update(distance)
                    
                    # Draw connection between index and thumb for right hand
                    cv2.line(frame, index_point, thumb_point, (0, 255, 255), 2)
                    cv2.circle(frame, index_point, 5, (0, 255, 255), -1)
                    cv2.circle(frame, thumb_point, 5, (0, 255, 255), -1)
                    
                    # Display volume at midpoint
                    mid_x = (index_point[0] + thumb_point[0]) // 2
                    mid_y = (index_point[1] + thumb_point[1]) // 2
                    draw_centered_label(frame, f"{int(current_volume * 100)}%", (mid_x, mid_y), size=0.6, thickness=2)
                    
                    # Volume bar
                    volume_bar_x = w - 80
                    volume_bar_y = (h - 200) // 2
                    volume_bar_h = 200
                    volume_bar_w = 30
                    
                    cv2.rectangle(frame, (volume_bar_x, volume_bar_y), 
                                (volume_bar_x + volume_bar_w, volume_bar_y + volume_bar_h), 
                                (200, 200, 200), -1)
                    
                    fill_h = int(volume_bar_h * current_volume)
                    cv2.rectangle(frame, (volume_bar_x, volume_bar_y + volume_bar_h - fill_h), 
                                (volume_bar_x + volume_bar_w, volume_bar_y + volume_bar_h),
                                (0, 255, 255), -1)
                    
                    draw_centered_label(frame, f"{int(current_volume * 100)}%", 
                                      (volume_bar_x + volume_bar_w // 2, volume_bar_y + volume_bar_h + 15), 0.5, 1)
                    
                    # Process volume control
                    current_time = time.time()
                    
                    if prev_right_hand_distance is not None:
                        distance_change = smoothed_distance - prev_right_hand_distance
                        
                        if abs(distance_change) > VOLUME_CHANGE_THRESHOLD:
                            volume_trend = 1 if distance_change > 0 else -1
                        else:
                            volume_trend = 0
                        
                        if current_time - last_volume_change > MIN_VOLUME_CHANGE_INTERVAL:
                            if abs(distance_change) > VOLUME_CHANGE_THRESHOLD:
                                direction = "louder" if distance_change > 0 else "quieter"
                                old_volume = current_volume
                                current_volume = adjust_volume(direction, distance_change)
                                
                                if current_volume != old_volume:
                                    last_volume_status = "Volume up" if current_volume > old_volume else "Volume down"
                                last_volume_change = current_time
                        
                        if volume_trend > 0:
                            draw_centered_label(frame, "🔊", 
                                             (volume_bar_x + volume_bar_w // 2, volume_bar_y - 15), 0.7, 2)
                        elif volume_trend < 0:
                            draw_centered_label(frame, "🔉", 
                                             (volume_bar_x + volume_bar_w // 2, volume_bar_y - 15), 0.7, 2)
                    
                    prev_right_hand_distance = smoothed_distance
                    
                    if last_volume_status:
                        color = (0, 255, 255)
                        draw_centered_label(frame, last_volume_status, 
                                         (volume_bar_x + volume_bar_w // 2, volume_bar_y - 35), 0.5, 1)
                
                # Display action status
                if action_status:
                    draw_centered_label(frame, action_status, (w // 2, 50), 0.6, 2)
                
                # Display FPS and status
                cv2.putText(frame, f"FPS: {fps}", (w - 80, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Using {browser_type.capitalize()}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                status_text = "Connected" if selenium_active else "Disconnected"
                status_color = (0, 255, 0) if selenium_active else (0, 0, 255)
                cv2.putText(frame, f"YouTube: {status_text}", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                cv2.imshow("Hand Controller", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    processing_active = False
            
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == 27:
                    processing_active = False
            except Exception as e:
                print(f"⚠️ Lỗi trong vòng lặp chính: {e}")
                log_gesture_result("System", False, 0, fps, f"Main loop error: {str(e)}", 0)
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        processing_active = False
    finally:
        processing_active = False
        # Write remaining logs to file
        if log_buffer:
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for log_entry in log_buffer:
                    writer.writerow(log_entry)
            log_buffer.clear()
        time.sleep(0.5)
        cv2.destroyAllWindows()
        if 'hands' in globals():
            hands.close()
        if driver:
            try:
                driver.quit()
            except:
                pass
        print(f"✅ Exited. Log saved to {log_file}")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        print("\nProgram terminated due to an error. Please check your setup and try again.")