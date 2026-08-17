import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import json
import os
from datetime import datetime
import platform
import subprocess
import threading


class UltraRobustGaitDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Get screen resolution
        import tkinter as tk
        root = tk.Tk()
        self.display_width = root.winfo_screenwidth()
        self.display_height = root.winfo_screenheight()
        root.destroy()

        # Voice/Audio feedback
        self.audio_enabled = True
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
            self.voice_available = True
        except:
            self.voice_available = False
            print("⚠️ Voice synthesis unavailable")

        # Calibration settings
        self.calibration_mode = False
        self.calibration_frames = 0
        self.calibration_duration = 60
        self.calibration_phase = 0
        self.last_phase_announced = -1
        self.last_progress_beep_second = -1  # Track last progress beep

        # ULTRA-ROBUST SETTINGS
        self.percentile_lower = 2
        self.percentile_upper = 98
        self.min_std_threshold = 5.0
        self.buffer_multiplier = 0.25
        self.std_multiplier = 3.0

        # Persistence settings
        self.persistence_window = 45
        self.persistence_threshold = 0.80
        self.min_persistence_frames = 20

        # Anomaly tracking
        self.anomaly_detected = False
        self.anomaly_just_detected = False

        # SMOOTHING
        self.smoothing_window = 5
        self.angle_buffer = {
            'left_knee': deque(maxlen=self.smoothing_window),
            'right_knee': deque(maxlen=self.smoothing_window),
            'left_hip': deque(maxlen=self.smoothing_window),
            'right_hip': deque(maxlen=self.smoothing_window)
        }

        # Store real-time data
        self.left_knee_angles = deque(maxlen=45)
        self.right_knee_angles = deque(maxlen=45)
        self.left_hip_angles = deque(maxlen=45)
        self.right_hip_angles = deque(maxlen=45)
        self.step_heights = deque(maxlen=45)

        # Persistence tracking
        self.anomaly_history = {
            'left_knee': deque(maxlen=self.persistence_window),
            'right_knee': deque(maxlen=self.persistence_window),
            'left_hip': deque(maxlen=self.persistence_window),
            'right_hip': deque(maxlen=self.persistence_window),
            'knee_asymmetry': deque(maxlen=self.persistence_window),
            'hip_asymmetry': deque(maxlen=self.persistence_window),
            'step_height': deque(maxlen=self.persistence_window),
            'rom_reduced': deque(maxlen=self.persistence_window)
        }

        self.frame_metrics = deque(maxlen=self.persistence_window)

        # Calibration data collection
        self.calibration_data = {
            'left_knee': [],
            'right_knee': [],
            'left_hip': [],
            'right_hip': [],
            'step_height': [],
            'knee_asymmetry': [],
            'hip_asymmetry': []
        }

        # Baseline
        self.baseline = None
        self.persistent_anomalies = []
        self.current_username = None

        # Debug mode
        self.debug_mode = True
        self.frame_count = 0

        # Sensitivity
        self.sensitivity_level = "Normal"
        self.apply_sensitivity_preset(self.sensitivity_level)

    def _play_beep_blocking(self, frequency=1000, duration=200):
        """Blocking beep implementation (runs in thread)"""
        try:
            system = platform.system()
            if system == 'Windows':
                import winsound
                winsound.Beep(frequency, duration)
            elif system == 'Darwin':
                os.system(f'afplay /System/Library/Sounds/Ping.aiff')
            elif system == 'Linux':
                try:
                    os.system(f'paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null')
                except:
                    print('\a')
            else:
                print('\a')
        except:
            print('\a')

    def play_beep(self, frequency=1000, duration=200):
        """Play a beep sound (non-blocking)"""
        if not self.audio_enabled:
            return
        threading.Thread(target=self._play_beep_blocking,
                         args=(frequency, duration),
                         daemon=True).start()

    def _double_beep(self):
        """Play double beep (blocking, runs in thread)"""
        self._play_beep_blocking(600, 200)
        import time
        time.sleep(0.15)
        self._play_beep_blocking(600, 200)

    def _triple_beep_ascending(self):
        """Play ascending beeps (blocking, runs in thread)"""
        import time
        self._play_beep_blocking(800, 150)
        time.sleep(0.1)
        self._play_beep_blocking(1000, 150)
        time.sleep(0.1)
        self._play_beep_blocking(1200, 300)

    def _countdown_beeps(self):
        """Play countdown beeps (blocking, runs in thread)"""
        import time
        for i in range(3, 0, -1):
            self._play_beep_blocking(800, 200)
            if self.voice_available:
                self._speak_blocking(str(i))
            time.sleep(1)

    def _speak_blocking(self, text):
        """Blocking TTS implementation"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except:
            pass

    def speak(self, text, wait=False):
        """Text-to-speech feedback (non-blocking by default)"""
        if self.voice_available:
            print(f"🔊 {text}")
            if wait:
                # Blocking speech (for initial setup)
                threading.Thread(target=self._speak_blocking,
                                 args=(text,),
                                 daemon=True).start()
            else:
                # Non-blocking speech
                threading.Thread(target=self._speak_blocking,
                                 args=(text,),
                                 daemon=True).start()
        else:
            print(f"💬 {text}")

    def check_calibration_phase(self):
        """Check calibration progress and provide audio cues"""
        elapsed_seconds = self.calibration_frames / 30.0
        current_second = int(elapsed_seconds)

        if elapsed_seconds < 20:
            phase = 1
        elif elapsed_seconds < 40:
            phase = 2
        else:
            phase = 3

        # Announce phase changes with double beeps (at 20s and 40s ONLY)
        if phase != self.last_phase_announced:
            self.calibration_phase = phase
            self.last_phase_announced = phase

            # Double beep for phase change (non-blocking)
            if phase == 2 or phase == 3:  # Only at 20s and 40s
                threading.Thread(target=self._double_beep, daemon=True).start()

            # Voice instruction
            if phase == 1:
                self.speak("Phase one. Walk normally at comfortable pace.", wait=True)
            elif phase == 2:
                self.speak("Phase two. Now walk slower, then faster.", wait=True)
            elif phase == 3:
                self.speak("Phase three. Now take bigger steps, then smaller steps.", wait=True)

        # Progress beeps at 10s, 30s, 50s ONLY (single beeps)
        # Make sure we only beep once per second using tracking variable
        if current_second in [10, 30, 50] and current_second != self.last_progress_beep_second:
            self.last_progress_beep_second = current_second
            self.play_beep(800, 150)

    def apply_sensitivity_preset(self, level):
        """Apply sensitivity presets"""
        if level == "Ultra-Low":
            self.std_multiplier = 4.0
            self.persistence_threshold = 0.90
            self.min_std_threshold = 8.0
            self.buffer_multiplier = 0.35
        elif level == "Low":
            self.std_multiplier = 3.5
            self.persistence_threshold = 0.85
            self.min_std_threshold = 6.0
            self.buffer_multiplier = 0.30
        elif level == "Normal":
            self.std_multiplier = 3.0
            self.persistence_threshold = 0.80
            self.min_std_threshold = 5.0
            self.buffer_multiplier = 0.25
        elif level == "High":
            self.std_multiplier = 2.5
            self.persistence_threshold = 0.70
            self.min_std_threshold = 3.0
            self.buffer_multiplier = 0.15

        self.sensitivity_level = level
        print(f"\n📊 Sensitivity: {level}")
        print(f"   Std multiplier: {self.std_multiplier}x")
        print(f"   Persistence: {self.persistence_threshold * 100:.0f}%")

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def smooth_angle(self, angle, joint_name):
        """Apply temporal smoothing to reduce noise"""
        self.angle_buffer[joint_name].append(angle)
        if len(self.angle_buffer[joint_name]) > 0:
            return np.mean(self.angle_buffer[joint_name])
        return angle

    def extract_landmarks(self, landmarks, image_shape):
        """Extract key body landmarks"""
        h, w = image_shape[:2]

        def get_coords(landmark):
            return [landmark.x * w, landmark.y * h]

        coords = {
            'left_hip': get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]),
            'right_hip': get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]),
            'left_knee': get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]),
            'right_knee': get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]),
            'left_ankle': get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]),
            'right_ankle': get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
            'left_shoulder': get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]),
            'right_shoulder': get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),
        }

        return coords

    def collect_calibration_data(self, coords):
        """Collect data during calibration phase"""
        self.check_calibration_phase()

        left_knee = self.calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
        right_knee = self.calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
        left_hip = self.calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
        right_hip = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])

        left_knee = self.smooth_angle(left_knee, 'left_knee')
        right_knee = self.smooth_angle(right_knee, 'right_knee')
        left_hip = self.smooth_angle(left_hip, 'left_hip')
        right_hip = self.smooth_angle(right_hip, 'right_hip')

        left_step = abs(coords['left_ankle'][1] - coords['left_hip'][1])
        right_step = abs(coords['right_ankle'][1] - coords['right_hip'][1])
        avg_step = (left_step + right_step) / 2

        self.calibration_data['left_knee'].append(left_knee)
        self.calibration_data['right_knee'].append(right_knee)
        self.calibration_data['left_hip'].append(left_hip)
        self.calibration_data['right_hip'].append(right_hip)
        self.calibration_data['step_height'].append(avg_step)
        self.calibration_data['knee_asymmetry'].append(abs(left_knee - right_knee))
        self.calibration_data['hip_asymmetry'].append(abs(left_hip - right_hip))

        self.calibration_frames += 1

    def calculate_baseline(self):
        """Calculate ultra-robust baseline"""
        baseline = {}

        for metric, values in self.calibration_data.items():
            if len(values) > 0:
                values_array = np.array(values)

                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                min_val = np.min(values_array)
                max_val = np.max(values_array)

                lower_bound = np.percentile(values_array, self.percentile_lower)
                upper_bound = np.percentile(values_array, self.percentile_upper)

                robust_std = max(std_val, self.min_std_threshold)

                std_lower = mean_val - (self.std_multiplier * robust_std)
                std_upper = mean_val + (self.std_multiplier * robust_std)

                percentile_range = upper_bound - lower_bound
                buffer = percentile_range * self.buffer_multiplier
                lower_bound -= buffer
                upper_bound += buffer

                final_lower = min(lower_bound, std_lower)
                final_upper = max(upper_bound, std_upper)

                baseline[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'robust_std': robust_std,
                    'min': min_val,
                    'max': max_val,
                    'samples': len(values),
                    'lower_bound': final_lower,
                    'upper_bound': final_upper,
                    'range_width': final_upper - final_lower,
                    'quality': self.assess_metric_quality(std_val, values_array)
                }

        return baseline

    def assess_metric_quality(self, std_val, values):
        """Assess quality of calibration data"""
        if std_val < 2.0:
            return "Very Low Variance (may need recalibration)"
        elif std_val < 4.0:
            return "Low Variance"
        elif std_val < 8.0:
            return "Good Variance"
        else:
            return "High Variance (very diverse)"

    def check_persistence(self, metric_name):
        """Check if anomaly persists"""
        if len(self.anomaly_history[metric_name]) < self.min_persistence_frames:
            return False

        anomalous_count = sum(self.anomaly_history[metric_name])
        total_count = len(self.anomaly_history[metric_name])
        persistence_rate = anomalous_count / total_count

        return persistence_rate >= self.persistence_threshold

    def is_value_anomalous(self, value, baseline_stats):
        """Check if value is outside the bounds"""
        lower = baseline_stats['lower_bound']
        upper = baseline_stats['upper_bound']
        return (value < lower) or (value > upper)

    def analyze_gait(self, coords):
        """Analyze gait with ultra-robust detection"""
        self.persistent_anomalies = []
        self.frame_count += 1

        left_knee_raw = self.calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
        right_knee_raw = self.calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
        left_hip_raw = self.calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
        right_hip_raw = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])

        left_knee_angle = self.smooth_angle(left_knee_raw, 'left_knee')
        right_knee_angle = self.smooth_angle(right_knee_raw, 'right_knee')
        left_hip_angle = self.smooth_angle(left_hip_raw, 'left_hip')
        right_hip_angle = self.smooth_angle(right_hip_raw, 'right_hip')

        self.left_knee_angles.append(left_knee_angle)
        self.right_knee_angles.append(right_knee_angle)
        self.left_hip_angles.append(left_hip_angle)
        self.right_hip_angles.append(right_hip_angle)

        left_step_height = abs(coords['left_ankle'][1] - coords['left_hip'][1])
        right_step_height = abs(coords['right_ankle'][1] - coords['right_hip'][1])
        avg_step_height = (left_step_height + right_step_height) / 2
        self.step_heights.append(avg_step_height)

        current_metrics = {
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle,
            'step_height': avg_step_height,
            'knee_asymmetry': abs(left_knee_angle - right_knee_angle),
            'hip_asymmetry': abs(left_hip_angle - right_hip_angle)
        }
        self.frame_metrics.append(current_metrics)

        if self.debug_mode and self.frame_count % 30 == 0 and self.baseline:
            print(f"\n[Frame {self.frame_count}] Left Knee:")
            print(f"  Current: {left_knee_angle:.1f}°")
            print(
                f"  Baseline: [{self.baseline['left_knee']['lower_bound']:.1f}° to {self.baseline['left_knee']['upper_bound']:.1f}°]")
            print(f"  Range width: {self.baseline['left_knee']['range_width']:.1f}°")
            is_outside = self.is_value_anomalous(left_knee_angle, self.baseline['left_knee'])
            print(f"  Status: {'❌ OUTSIDE' if is_outside else '✅ INSIDE'}")

        if self.baseline:
            is_lk_anomalous = self.is_value_anomalous(left_knee_angle, self.baseline['left_knee'])
            self.anomaly_history['left_knee'].append(is_lk_anomalous)

            is_rk_anomalous = self.is_value_anomalous(right_knee_angle, self.baseline['right_knee'])
            self.anomaly_history['right_knee'].append(is_rk_anomalous)

            is_lh_anomalous = self.is_value_anomalous(left_hip_angle, self.baseline['left_hip'])
            self.anomaly_history['left_hip'].append(is_lh_anomalous)

            is_rh_anomalous = self.is_value_anomalous(right_hip_angle, self.baseline['right_hip'])
            self.anomaly_history['right_hip'].append(is_rh_anomalous)

            knee_asym = abs(left_knee_angle - right_knee_angle)
            is_ka_anomalous = self.is_value_anomalous(knee_asym, self.baseline['knee_asymmetry'])
            self.anomaly_history['knee_asymmetry'].append(is_ka_anomalous)

            hip_asym = abs(left_hip_angle - right_hip_angle)
            is_ha_anomalous = self.is_value_anomalous(hip_asym, self.baseline['hip_asymmetry'])
            self.anomaly_history['hip_asymmetry'].append(is_ha_anomalous)

            is_sh_anomalous = self.is_value_anomalous(avg_step_height, self.baseline['step_height'])
            self.anomaly_history['step_height'].append(is_sh_anomalous)

            # ROM check - compare recent range to baseline MEAN range (not max)
            is_rom_anomalous = False
            if len(self.frame_metrics) >= 30:  # Need more frames for reliable ROM
                recent_lk = [m['left_knee'] for m in list(self.frame_metrics)[-30:]]
                recent_rk = [m['right_knee'] for m in list(self.frame_metrics)[-30:]]
                lk_range = max(recent_lk) - min(recent_lk)
                rk_range = max(recent_rk) - min(recent_rk)

                # Calculate average range during calibration (more realistic than max)
                baseline_ranges = []
                for i in range(0, len(self.calibration_data['left_knee']) - 30, 15):
                    window = self.calibration_data['left_knee'][i:i + 30]
                    if len(window) == 30:
                        baseline_ranges.append(max(window) - min(window))

                if len(baseline_ranges) > 0:
                    avg_baseline_rom = np.mean(baseline_ranges)
                    # Only flag if BOTH knees have very reduced ROM (20% of average)
                    if lk_range < avg_baseline_rom * 0.20 and rk_range < avg_baseline_rom * 0.20:
                        is_rom_anomalous = True

            self.anomaly_history['rom_reduced'].append(is_rom_anomalous)

            if self.check_persistence('left_knee'):
                deviation = left_knee_angle - self.baseline['left_knee']['mean']
                persistence_rate = sum(self.anomaly_history['left_knee']) / len(self.anomaly_history['left_knee'])
                self.persistent_anomalies.append(
                    f"Left Knee: {deviation:+.1f}° ({persistence_rate * 100:.0f}%)"
                )

            if self.check_persistence('right_knee'):
                deviation = right_knee_angle - self.baseline['right_knee']['mean']
                persistence_rate = sum(self.anomaly_history['right_knee']) / len(self.anomaly_history['right_knee'])
                self.persistent_anomalies.append(
                    f"Right Knee: {deviation:+.1f}° ({persistence_rate * 100:.0f}%)"
                )

            if self.check_persistence('knee_asymmetry'):
                persistence_rate = sum(self.anomaly_history['knee_asymmetry']) / len(
                    self.anomaly_history['knee_asymmetry'])
                self.persistent_anomalies.append(
                    f"Knee Asymmetry: {knee_asym:.1f}° ({persistence_rate * 100:.0f}%)"
                )

            if self.check_persistence('hip_asymmetry'):
                persistence_rate = sum(self.anomaly_history['hip_asymmetry']) / len(
                    self.anomaly_history['hip_asymmetry'])
                self.persistent_anomalies.append(
                    f"Hip Asymmetry: {hip_asym:.1f}° ({persistence_rate * 100:.0f}%)"
                )

            # ROM check disabled - too sensitive for real-world use
            # if self.check_persistence('rom_reduced'):
            #     persistence_rate = sum(self.anomaly_history['rom_reduced']) / len(self.anomaly_history['rom_reduced'])
            #     self.persistent_anomalies.append(
            #         f"Reduced Mobility ({persistence_rate * 100:.0f}%)"
            #     )

        # Audio alert when anomalies are first detected (NON-BLOCKING)
        if len(self.persistent_anomalies) > 0 and not self.anomaly_detected:
            self.anomaly_detected = True
            self.anomaly_just_detected = True

            # Double beep (non-blocking)
            threading.Thread(target=self._double_beep, daemon=True).start()

            # Voice alert (non-blocking)
            if len(self.persistent_anomalies) == 1:
                self.speak(f"Gait anomaly detected: {self.persistent_anomalies[0]}", wait=False)
            else:
                self.speak(f"Multiple gait anomalies detected. Please check the screen.", wait=False)

        elif len(self.persistent_anomalies) == 0 and self.anomaly_detected:
            self.anomaly_detected = False

            # Ascending beeps (non-blocking)
            threading.Thread(target=self._triple_beep_ascending, daemon=True).start()

            self.speak("Gait has returned to normal.", wait=False)

        return {
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle
        }

    def draw_annotations(self, image, coords, angles):
        """Draw annotations with phase indicators during calibration"""
        h, w = image.shape[:2]

        if self.calibration_mode:
            elapsed_seconds = self.calibration_frames / 30.0
            progress = (self.calibration_frames / (self.calibration_duration * 30)) * 100

            cv2.rectangle(image, (0, 0), (w, 200), (50, 50, 50), -1)

            cv2.putText(image, "CALIBRATION MODE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.putText(image, f"Progress: {progress:.1f}% ({elapsed_seconds:.0f}s / 60s)", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            phase_text = ""
            phase_color = (255, 255, 255)

            if elapsed_seconds < 20:
                phase_text = "PHASE 1: Walk normally at comfortable pace"
                phase_color = (0, 255, 0)
            elif elapsed_seconds < 40:
                phase_text = "PHASE 2: Walk slower, then faster"
                phase_color = (0, 255, 255)
            else:
                phase_text = "PHASE 3: Take bigger steps, then smaller steps"
                phase_color = (255, 0, 255)

            cv2.putText(image, phase_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

            bar_y = 150
            bar_height = 30

            phase1_color = (0, 200, 0) if elapsed_seconds < 20 else (0, 100, 0)
            cv2.rectangle(image, (10, bar_y), (10 + int(w / 3) - 10, bar_y + bar_height), phase1_color, -1)
            cv2.putText(image, "1", (10 + int(w / 6) - 10, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            phase2_color = (0, 200, 200) if 20 <= elapsed_seconds < 40 else (0, 100, 100)
            cv2.rectangle(image, (10 + int(w / 3), bar_y), (10 + int(2 * w / 3) - 10, bar_y + bar_height), phase2_color,
                          -1)
            cv2.putText(image, "2", (10 + int(w / 2) - 10, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            phase3_color = (200, 0, 200) if elapsed_seconds >= 40 else (100, 0, 100)
            cv2.rectangle(image, (10 + int(2 * w / 3), bar_y), (w - 10, bar_y + bar_height), phase3_color, -1)
            cv2.putText(image, "3", (10 + int(5 * w / 6) - 10, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            y_offset = 30
            if self.baseline:
                cv2.putText(image, f"Sensitivity: {self.sensitivity_level}", (w - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                if self.persistent_anomalies:
                    cv2.putText(image, "PERSISTENT ANOMALIES:", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30

                    for anomaly in self.persistent_anomalies[:5]:
                        cv2.putText(image, f"- {anomaly}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        y_offset += 25
                else:
                    cv2.putText(image, "Gait: NORMAL", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                    y_offset += 35
                    cv2.putText(image, "(No sustained deviations)", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if self.debug_mode:
                    y_offset += 50
                    lk_stats = self.baseline['left_knee']
                    current_lk = angles['left_knee']

                    cv2.putText(image, "--- DEBUG INFO ---", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y_offset += 25

                    cv2.putText(image, f"Left Knee: {current_lk:.1f}°", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

                    cv2.putText(image, f"Allowed: [{lk_stats['lower_bound']:.1f}° to {lk_stats['upper_bound']:.1f}°]",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 20

                    cv2.putText(image, f"Range width: {lk_stats['range_width']:.1f}°",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 20

                    is_outside = self.is_value_anomalous(current_lk, lk_stats)
                    status_color = (0, 0, 255) if is_outside else (0, 255, 0)
                    status_text = "OUTSIDE bounds" if is_outside else "INSIDE bounds"
                    cv2.putText(image, f"Status: {status_text}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                    if len(self.anomaly_history['left_knee']) > 0:
                        y_offset += 25
                        anomaly_count = sum(self.anomaly_history['left_knee'])
                        total = len(self.anomaly_history['left_knee'])
                        percent = (anomaly_count / total) * 100 if total > 0 else 0
                        cv2.putText(image, f"Anomalous: {anomaly_count}/{total} ({percent:.0f}%)",
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        y_offset += 20
                        cv2.putText(image, f"Need {self.persistence_threshold * 100:.0f}% to flag",
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(image, "NO BASELINE", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(image, "Press 'C' to calibrate", (10, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return image

    def start_calibration(self, username=None):
        """Start calibration with countdown and audio instructions"""
        self.calibration_mode = True
        self.calibration_frames = 0
        self.current_username = username
        self.calibration_phase = 0
        self.last_phase_announced = -1
        self.last_progress_beep_second = -1  # Reset progress beep tracker

        self.calibration_data = {
            'left_knee': [], 'right_knee': [], 'left_hip': [],
            'right_hip': [], 'step_height': [], 'knee_asymmetry': [],
            'hip_asymmetry': []
        }

        for key in self.angle_buffer:
            self.angle_buffer[key].clear()

        print("\n" + "=" * 70)
        print("🎯 CALIBRATION STARTING")
        print("=" * 70)
        print("\n⏱️  60-SECOND CALIBRATION WITH 3 PHASES:\n")
        print("   🚶 PHASE 1 (0-20s):  Walk normally at comfortable pace")
        print("   ⚡ PHASE 2 (20-40s): Walk slower, then faster")
        print("   📏 PHASE 3 (40-60s): Take bigger steps, then smaller steps")
        print("\n🔊 You will hear BEEPS and VOICE INSTRUCTIONS for each phase!")
        print("=" * 70)

        self.speak("Calibration will start in 3 seconds. Get ready.", wait=True)

        # Countdown in separate thread (non-blocking)
        threading.Thread(target=self._countdown_beeps, daemon=True).start()

        # Wait 3 seconds for countdown (but don't block main thread completely)
        import time
        time.sleep(3)

        self.play_beep(1500, 300)
        print("\n✅ CALIBRATION STARTED!\n")

    def finish_calibration(self):
        """Finish calibration with completion sound"""
        self.calibration_mode = False
        self.baseline = self.calculate_baseline()

        # Completion sound (non-blocking)
        threading.Thread(target=self._triple_beep_ascending, daemon=True).start()

        self.speak("Calibration complete!", wait=True)

        print("\n" + "=" * 70)
        print("✅ CALIBRATION COMPLETE!")
        print("=" * 70)

        print("\n📊 Your Baseline Ranges:")
        for metric in ['left_knee', 'right_knee', 'knee_asymmetry']:
            stats = self.baseline[metric]
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.1f}°")
            print(f"  Actual std: {stats['std']:.1f}°")
            print(f"  Enforced std: {stats['robust_std']:.1f}°")
            print(f"  Allowed range: [{stats['lower_bound']:.1f}° to {stats['upper_bound']:.1f}°]")
            print(f"  Range width: {stats['range_width']:.1f}°")
            print(f"  Quality: {stats['quality']}")

        print(f"\n🎚️  Detection Settings (Sensitivity: {self.sensitivity_level}):")
        print(f"  Std multiplier: {self.std_multiplier}x")
        print(f"  Persistence required: {self.persistence_threshold * 100:.0f}%")
        print(f"  Minimum frames: {self.min_persistence_frames}")

        if self.current_username:
            self.save_baseline(self.current_username)

        print("=" * 70 + "\n")

        self.speak("You may now begin walking normally. I will monitor for any persistent changes.", wait=True)

    def save_baseline(self, username, filename="gait_profiles.json"):
        """Save baseline"""
        if not self.baseline:
            return

        profiles = {}
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                profiles = json.load(f)

        profiles[username] = {
            'baseline': self.baseline,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sensitivity': self.sensitivity_level,
            'settings': {
                'std_multiplier': self.std_multiplier,
                'persistence_threshold': self.persistence_threshold,
                'min_std_threshold': self.min_std_threshold
            }
        }

        with open(filename, 'w') as f:
            json.dump(profiles, f, indent=2)

        print(f"💾 Profile saved: {username}")

    def load_baseline(self, username, filename="gait_profiles.json"):
        """Load baseline"""
        if not os.path.exists(filename):
            print(f"No profile file found")
            return False

        with open(filename, 'r') as f:
            profiles = json.load(f)

        if username not in profiles:
            print(f"No profile for: {username}")
            return False

        self.baseline = profiles[username]['baseline']
        if 'sensitivity' in profiles[username]:
            self.sensitivity_level = profiles[username]['sensitivity']
            if 'settings' in profiles[username]:
                settings = profiles[username]['settings']
                self.std_multiplier = settings.get('std_multiplier', 3.0)
                self.persistence_threshold = settings.get('persistence_threshold', 0.80)

        print(f"✅ Profile loaded: {username}")
        print(f"   Sensitivity: {self.sensitivity_level}")
        return True

    def process_frame(self, frame):
        """Process frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            coords = self.extract_landmarks(results.pose_landmarks.landmark, frame.shape)

            if self.calibration_mode:
                self.collect_calibration_data(coords)
                if self.calibration_frames >= (self.calibration_duration * 30):
                    self.finish_calibration()
                angles = {
                    'left_knee': self.calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle']),
                    'right_knee': self.calculate_angle(coords['right_hip'], coords['right_knee'],
                                                       coords['right_ankle']),
                    'left_hip': 0, 'right_hip': 0
                }
            else:
                angles = self.analyze_gait(coords)

            frame = self.draw_annotations(frame, coords, angles)

        return frame

    def resize_to_fit_screen(self, frame):
        """Resize frame to fit screen"""
        h, w = frame.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        y_offset = (self.display_height - new_h) // 2
        x_offset = (self.display_width - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def run_video(self, video_path=0, username=None):
        """Run gait analysis"""
        cap = cv2.VideoCapture(video_path)

        if video_path == 0 or isinstance(video_path, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            cap.set(cv2.CAP_PROP_FPS, 120)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera: {actual_width}x{actual_height} @ {actual_fps}fps")

        if username and not self.baseline:
            self.load_baseline(username)

        window_name = 'Ultra-Robust Gait Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)

        print("\n" + "=" * 60)
        print("ULTRA-ROBUST GAIT DETECTION")
        print("=" * 60)
        print(f"Current Sensitivity: {self.sensitivity_level}")
        print(f"🔊 Audio Feedback: {'Enabled' if self.audio_enabled else 'Disabled'}")
        print("\n🎵 During calibration you will hear:")
        print("   • Countdown beeps (3-2-1)")
        print("   • Phase change beeps (double beep)")
        print("   • Progress beeps (every 10 seconds)")
        print("   • Voice instructions for each phase")
        print("\nControls:")
        print("  'c' - Start calibration (with audio guidance)")
        print("  's' - Save profile")
        print("  'l' - Load profile")
        print("  'd' - Toggle debug")
        print("  'f' - Fullscreen")
        print("  'a' - Toggle audio on/off")
        print("  '1' - Ultra-Low sensitivity (least false positives)")
        print("  '2' - Low sensitivity")
        print("  '3' - Normal sensitivity (default)")
        print("  '4' - High sensitivity (catch subtle changes)")
        print("  'q' - Quit")
        print("=" * 60 + "\n")

        fullscreen = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            display_frame = self.resize_to_fit_screen(frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and not self.calibration_mode:
                username_input = input("\nUsername: ").strip()
                self.start_calibration(username_input if username_input else None)
            elif key == ord('s') and self.baseline:
                user = input("\nSave as: ")
                self.save_baseline(user)
            elif key == ord('l'):
                user = input("\nLoad: ")
                self.load_baseline(user)
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('a'):
                self.audio_enabled = not self.audio_enabled
                print(f"Audio: {'ON' if self.audio_enabled else 'OFF'}")
                if self.audio_enabled:
                    self.play_beep(1000, 200)
            elif key == ord('f'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('1'):
                self.apply_sensitivity_preset("Ultra-Low")
            elif key == ord('2'):
                self.apply_sensitivity_preset("Low")
            elif key == ord('3'):
                self.apply_sensitivity_preset("Normal")
            elif key == ord('4'):
                self.apply_sensitivity_preset("High")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = UltraRobustGaitDetector()
    detector.run_video(0, username="test_user")

    # Use a video file instead of camera
# detector.run('path/to/video.mp4')