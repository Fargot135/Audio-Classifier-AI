"""
Tkinter GUI for Audio Classifier
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sounddevice as sd
import numpy as np

from audio.processor import AudioProcessor
from config import DEVICE, DURATION, SAMPLE_RATE, SILENCE_THRESHOLD


class AudioClassifierGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Audio Classifier")
        self.root.geometry("600x500")
        self.root.configure(bg='#2b2b2b')
        
        # State
        self.is_running = False
        self.recording_thread = None
        
        # Audio processor
        self.processor = AudioProcessor()
        if not self.processor.load_model():
            messagebox.showerror("Error", "Model file 'sound_model.pth' not found!")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create GUI elements"""
        # Title
        title = tk.Label(
            self.root,
            text="🎵 Real-time Audio Classifier",
            font=("Arial", 24, "bold"),
            bg='#2b2b2b',
            fg='#ffa500'
        )
        title.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to listen...",
            font=("Arial", 14),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        self.status_label.pack(pady=10)
        
        # Result frame
        result_frame = tk.Frame(self.root, bg='#1a1a1a', relief=tk.RAISED, borderwidth=2)
        result_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Result label
        self.result_label = tk.Label(
            result_frame,
            text="---",
            font=("Arial", 48, "bold"),
            bg='#1a1a1a',
            fg='#00ff00'
        )
        self.result_label.pack(pady=20)
        
        # Confidence label
        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: ---%",
            font=("Arial", 16),
            bg='#1a1a1a',
            fg='#aaaaaa'
        )
        self.confidence_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='determinate',
            length=400,
            maximum=100
        )
        self.progress.pack(pady=10)
        
        # Timer label
        self.timer_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#aaaaaa'
        )
        self.timer_label.pack()
        
        # Buttons frame
        btn_frame = tk.Frame(self.root, bg='#2b2b2b')
        btn_frame.pack(pady=20)
        
        # Start button
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ START",
            command=self.start_listening,
            font=("Arial", 14, "bold"),
            bg='#00aa00',
            fg='white',
            width=12,
            height=2,
            relief=tk.RAISED,
            cursor="hand2"
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        # Stop button
        self.stop_btn = tk.Button(
            btn_frame,
            text="⏸ STOP",
            command=self.stop_listening,
            font=("Arial", 14, "bold"),
            bg='#aa0000',
            fg='white',
            width=12,
            height=2,
            relief=tk.RAISED,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Device info
        device_info = tk.Label(
            self.root,
            text=f"Device: {DEVICE} | Duration: {DURATION}s",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#888888'
        )
        device_info.pack(side=tk.BOTTOM, pady=10)
    
    def start_listening(self):
        """Start audio recording loop"""
        if self.processor.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="🎤 Listening...", fg='#00ff00')
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()
    
    def stop_listening(self):
        """Stop audio recording"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏸ Stopped", fg='#ff6600')
        self.progress['value'] = 0
        self.timer_label.config(text="")
        self.result_label.config(text="---", fg='#00ff00')
        self.confidence_label.config(text="Confidence: ---%")
    
    def recording_loop(self):
        """Main recording loop (runs in thread)"""
        while self.is_running:
            try:
                # Reset progress
                self.root.after(0, lambda: self.progress.config(value=0))
                self.root.after(0, lambda: self.timer_label.config(text="🎤 Recording... 0.0s"))
                
                # Start recording in background
                num_samples = int(DURATION * SAMPLE_RATE)
                recording = sd.rec(num_samples, samplerate=SAMPLE_RATE, channels=1)
                
                # Animate progress bar smoothly while recording
                start_time = sd.get_stream().time
                update_interval = 0.05  # Update every 50ms for smooth animation
                
                while True:
                    if not self.is_running:
                        sd.stop()
                        return
                    
                    # Calculate elapsed time
                    elapsed = sd.get_stream().time - start_time
                    
                    if elapsed >= DURATION:
                        break
                    
                    # Update progress smoothly
                    progress_pct = (elapsed / DURATION) * 100
                    self.root.after(0, lambda p=progress_pct, e=elapsed: self.update_progress(p, e))
                    
                    # Small delay for smooth animation
                    sd.sleep(int(update_interval * 1000))
                
                # Wait for recording to finish
                sd.wait()
                
                # Set to 100%
                self.root.after(0, lambda: self.update_progress(100, DURATION))
                
                # Show processing
                self.root.after(0, lambda: self.timer_label.config(text="⚙️ Processing..."))
                
                # Check if silent
                max_amp = np.max(np.abs(recording))
                
                if max_amp > SILENCE_THRESHOLD:
                    # Normalize
                    recording = recording / (max_amp + 1e-6)
                    
                    # Predict
                    result, confidence = self.processor.predict_audio(recording)
                    
                    # Update GUI
                    self.root.after(0, self.update_result, result, confidence)
                else:
                    self.root.after(0, self.update_silence)
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Recording error: {e}"))
                self.is_running = False
    
    def update_progress(self, percent, elapsed):
        """Update progress bar and timer"""
        self.progress['value'] = percent
        self.timer_label.config(text=f"🎤 Recording... {elapsed:.1f}s / {DURATION:.1f}s")
    
    def update_result(self, result, confidence):
        """Update GUI with prediction result"""
        color = '#00ff00' if result == 'music' else '#ff9900'
        self.result_label.config(text=result.upper(), fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        self.progress['value'] = 0
        self.timer_label.config(text="✅ Done! Waiting for next recording...")
    
    def update_silence(self):
        """Update GUI for silence"""
        self.result_label.config(text="🔇 SILENCE", fg='#666666')
        self.confidence_label.config(text="Below threshold")
        self.progress['value'] = 0
        self.timer_label.config(text="✅ Done! Waiting for next recording...")