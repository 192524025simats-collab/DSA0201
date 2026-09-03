import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time


class HumanMotionAnalysis:

    def __init__(self, root):

        self.root = root
        self.root.title("Human Motion Analysis System")
        self.root.geometry("1250x780")
        self.root.configure(bg="#17202A")

        # =====================================================
        # VIDEO VARIABLES
        # =====================================================

        self.cap = None
        self.video_path = None
        self.output_writer = None

        self.video_fps = 30.0
        self.video_width = 0
        self.video_height = 0

        self.running = False
        self.paused = False

        # =====================================================
        # MOTION VARIABLES
        # =====================================================

        self.prev_gray = None
        self.prev_center = None

        self.total_frames = 0
        self.processed_frames = 0
        self.motion_frames = 0

        self.total_displacement = 0.0
        self.max_displacement = 0.0

        self.motion_history = []
        self.frame_history = []

        self.start_time = None
        self.latest_frame = None

        # =====================================================
        # CREATE FOLDERS
        # =====================================================

        os.makedirs("output", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

        # =====================================================
        # BUILD GUI
        # =====================================================

        self.build_gui()

    # =========================================================
    # BUILD GUI
    # =========================================================

    def build_gui(self):

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg="#1F618D",
            height=80
        )

        header.pack(
            fill="x"
        )

        tk.Label(
            header,
            text="REAL-TIME HUMAN MOTION ANALYSIS",
            font=("Arial", 23, "bold"),
            bg="#1F618D",
            fg="white"
        ).pack(
            pady=(12, 2)
        )

        tk.Label(
            header,
            text="Computer Vision • Video Processing • Motion Computation • Geometry",
            font=("Arial", 11),
            bg="#1F618D",
            fg="white"
        ).pack()

        # -----------------------------------------------------
        # MAIN FRAME
        # -----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg="#17202A"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        # =====================================================
        # VIDEO PANEL
        # =====================================================

        video_panel = tk.Frame(
            main,
            bg="#212F3D",
            bd=2,
            relief="ridge"
        )

        video_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        tk.Label(
            video_panel,
            text="VIDEO ANALYSIS",
            font=("Arial", 15, "bold"),
            bg="#212F3D",
            fg="white"
        ).pack(
            pady=8
        )

        self.video_label = tk.Label(
            video_panel,
            text="No video selected\n\nClick LOAD VIDEO",
            font=("Arial", 15),
            bg="#0B0F12",
            fg="#AAB7B8"
        )

        self.video_label.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=8
        )

        # -----------------------------------------------------
        # CONTROL BUTTONS
        # -----------------------------------------------------

        controls = tk.Frame(
            video_panel,
            bg="#212F3D"
        )

        controls.pack(
            pady=10
        )

        self.create_button(
            controls,
            "LOAD VIDEO",
            self.load_video,
            0
        )

        self.create_button(
            controls,
            "START",
            self.start_video,
            1
        )

        self.create_button(
            controls,
            "PAUSE",
            self.pause_video,
            2
        )

        self.create_button(
            controls,
            "STOP",
            self.stop_video,
            3
        )

        # =====================================================
        # RIGHT ANALYSIS PANEL
        # =====================================================

        panel = tk.Frame(
            main,
            bg="#212F3D",
            width=350,
            bd=2,
            relief="ridge"
        )

        panel.pack(
            side="right",
            fill="y"
        )

        panel.pack_propagate(False)

        tk.Label(
            panel,
            text="ANALYSIS RESULTS",
            font=("Arial", 16, "bold"),
            bg="#212F3D",
            fg="white"
        ).pack(
            pady=12
        )

        self.status = tk.Label(
            panel,
            text="Status: Waiting",
            font=("Arial", 11, "bold"),
            bg="#212F3D",
            fg="#F7DC6F"
        )

        self.status.pack(
            pady=5
        )

        # -----------------------------------------------------
        # METRICS
        # -----------------------------------------------------

        self.create_metric(
            panel,
            "Video",
            "video"
        )

        self.create_metric(
            panel,
            "Total Frames",
            "total"
        )

        self.create_metric(
            panel,
            "Processed Frames",
            "processed"
        )

        self.create_metric(
            panel,
            "Motion Frames",
            "motion"
        )

        self.create_metric(
            panel,
            "Current Displacement",
            "current"
        )

        self.create_metric(
            panel,
            "Total Displacement",
            "distance"
        )

        self.create_metric(
            panel,
            "Maximum Displacement",
            "maximum"
        )

        self.create_metric(
            panel,
            "Processing FPS",
            "processing_fps"
        )

        # -----------------------------------------------------
        # ANALYSIS BUTTONS
        # -----------------------------------------------------

        tk.Button(
            panel,
            text="MOTION GRAPH",
            command=self.show_graph,
            font=("Arial", 10, "bold"),
            width=25
        ).pack(
            pady=(20, 5)
        )

        tk.Button(
            panel,
            text="SAVE CURRENT FRAME",
            command=self.save_frame,
            font=("Arial", 10, "bold"),
            width=25
        ).pack(
            pady=5
        )

        tk.Button(
            panel,
            text="RESET",
            command=self.reset,
            font=("Arial", 10, "bold"),
            width=25
        ).pack(
            pady=5
        )

        # =====================================================
        # FOOTER
        # =====================================================

        tk.Label(
            self.root,
            text="CO4: Computer Vision Application     |     CO5: Video Processing, Motion & Geometry     |     SDG 9",
            font=("Arial", 10, "bold"),
            bg="#1F618D",
            fg="white"
        ).pack(
            fill="x",
            ipady=8
        )

    # =========================================================
    # CREATE BUTTON
    # =========================================================

    def create_button(
        self,
        parent,
        text,
        command,
        column
    ):

        tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 10, "bold"),
            width=13
        ).grid(
            row=0,
            column=column,
            padx=4
        )

    # =========================================================
    # CREATE METRIC
    # =========================================================

    def create_metric(
        self,
        parent,
        title,
        name
    ):

        frame = tk.Frame(
            parent,
            bg="#2C3E50"
        )

        frame.pack(
            fill="x",
            padx=15,
            pady=4
        )

        tk.Label(
            frame,
            text=title,
            font=("Arial", 9),
            bg="#2C3E50",
            fg="#D5D8DC"
        ).pack(
            side="left",
            padx=8,
            pady=7
        )

        value = tk.Label(
            frame,
            text="0",
            font=("Arial", 9, "bold"),
            bg="#2C3E50",
            fg="white"
        )

        value.pack(
            side="right",
            padx=8
        )

        setattr(
            self,
            name + "_label",
            value
        )

    # =========================================================
    # LOAD VIDEO
    # =========================================================

    def load_video(self):

        path = filedialog.askopenfilename(
            title="Select Human Motion Video",
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("MOV Video", "*.mov"),
                ("MKV Video", "*.mkv"),
                ("All Video Files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )

        if not path:
            return

        self.video_path = path

        # Release previous video
        if self.cap is not None:
            self.cap.release()

        if self.output_writer is not None:
            self.output_writer.release()

        # Open video
        self.cap = cv2.VideoCapture(
            self.video_path
        )

        if not self.cap.isOpened():

            messagebox.showerror(
                "Error",
                "Unable to open the selected video."
            )

            return

        # -----------------------------------------------------
        # Read video properties
        # -----------------------------------------------------

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.video_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.video_fps <= 0:
            self.video_fps = 30.0

        self.video_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.video_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        # -----------------------------------------------------
        # Create output writer
        # -----------------------------------------------------

        output_path = os.path.join(
            "output",
            "processed_human_motion.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.output_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.video_fps,
            (
                self.video_width,
                self.video_height
            )
        )

        # -----------------------------------------------------
        # Read first frame
        # -----------------------------------------------------

        ret, frame = self.cap.read()

        if ret:
            self.display_frame(frame)

        # Reset video position
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        # -----------------------------------------------------
        # Update GUI
        # -----------------------------------------------------

        filename = os.path.basename(
            self.video_path
        )

        self.video_label_name = filename

        self.video_label_info(
            filename
        )

        self.status.config(
            text="Status: Video Loaded",
            fg="#58D68D"
        )

    # =========================================================
    # UPDATE VIDEO LABEL
    # =========================================================

    def video_label_info(
        self,
        filename
    ):

        self.video_value_label.config(
            text=filename
        )

        self.total_label.config(
            text=str(
                self.total_frames
            )
        )

    # =========================================================
    # START VIDEO
    # =========================================================

    def start_video(self):

        if self.cap is None:

            messagebox.showwarning(
                "Warning",
                "Please load a video first."
            )

            return

        # If video already finished, restart
        current_frame = int(
            self.cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame >= self.total_frames:

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            self.prev_gray = None
            self.prev_center = None

        self.running = True
        self.paused = False

        self.status.config(
            text="Status: Processing",
            fg="#58D68D"
        )

        self.start_time = time.time()

        self.process_frame()

    # =========================================================
    # PAUSE VIDEO
    # =========================================================

    def pause_video(self):

        if self.cap is None:
            return

        self.paused = True

        self.status.config(
            text="Status: Paused",
            fg="#F7DC6F"
        )

    # =========================================================
    # STOP VIDEO
    # =========================================================

    def stop_video(self):

        self.running = False
        self.paused = False

        self.status.config(
            text="Status: Stopped",
            fg="#EC7063"
        )

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def process_frame(self):

        if not self.running:
            return

        # -----------------------------------------------------
        # Pause
        # -----------------------------------------------------

        if self.paused:

            self.root.after(
                100,
                self.process_frame
            )

            return

        # -----------------------------------------------------
        # Read frame
        # -----------------------------------------------------

        ret, frame = self.cap.read()

        if not ret:

            self.running = False

            if self.output_writer is not None:
                self.output_writer.release()
                self.output_writer = None

            self.status.config(
                text="Status: Completed",
                fg="#58D68D"
            )

            messagebox.showinfo(
                "Processing Completed",
                "Video processing completed.\n\n"
                "Processed video saved at:\n"
                "output/processed_human_motion.mp4"
            )

            return

        self.processed_frames += 1

        output_frame = frame.copy()

        # =====================================================
        # GRAYSCALE CONVERSION
        # =====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )

        motion_area = 0
        displacement = 0.0

        # =====================================================
        # MOTION DETECTION
        # =====================================================

        if self.prev_gray is not None:

            difference = cv2.absdiff(
                self.prev_gray,
                gray
            )

            _, threshold = cv2.threshold(
                difference,
                25,
                255,
                cv2.THRESH_BINARY
            )

            threshold = cv2.dilate(
                threshold,
                None,
                iterations=2
            )

            contours, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            largest = None
            largest_area = 0

            # -------------------------------------------------
            # Find moving regions
            # -------------------------------------------------

            for contour in contours:

                area = cv2.contourArea(
                    contour
                )

                # Ignore small noise
                if area < 500:
                    continue

                x, y, w, h = cv2.boundingRect(
                    contour
                )

                # Center of bounding box
                center = (
                    x + w // 2,
                    y + h // 2
                )

                # Draw detected region
                cv2.rectangle(
                    output_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    output_frame,
                    center,
                    5,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    output_frame,
                    "Motion Region",
                    (
                        x,
                        max(
                            20,
                            y - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # Select largest region
                if area > largest_area:

                    largest_area = area

                    largest = (
                        x,
                        y,
                        w,
                        h,
                        center
                    )

                motion_area += area

            # =================================================
            # GEOMETRIC ANALYSIS
            # =================================================

            if largest is not None:

                (
                    x,
                    y,
                    w,
                    h,
                    center
                ) = largest

                # Highlight primary moving object
                cv2.rectangle(
                    output_frame,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    (0, 0, 255),
                    3
                )

                # Width and height
                cv2.putText(
                    output_frame,
                    f"Width: {w}px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    output_frame,
                    f"Height: {h}px",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    output_frame,
                    f"Center: ({center[0]}, {center[1]})",
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Calculate displacement
                # ---------------------------------------------

                if self.prev_center is not None:

                    dx = (
                        center[0]
                        - self.prev_center[0]
                    )

                    dy = (
                        center[1]
                        - self.prev_center[1]
                    )

                    displacement = math.sqrt(
                        (dx ** 2) +
                        (dy ** 2)
                    )

                    self.total_displacement += (
                        displacement
                    )

                    if displacement > self.max_displacement:

                        self.max_displacement = (
                            displacement
                        )

                self.prev_center = center

                cv2.putText(
                    output_frame,
                    f"Displacement: {displacement:.2f}px",
                    (10, 114),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

        # =====================================================
        # MOTION CLASSIFICATION
        # =====================================================

        if motion_area > 1000:

            self.motion_frames += 1

            cv2.putText(
                output_frame,
                "MOTION DETECTED",
                (
                    10,
                    self.video_height - 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                output_frame,
                "NO SIGNIFICANT MOTION",
                (
                    10,
                    self.video_height - 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

        # =====================================================
        # STORE HISTORY
        # =====================================================

        self.frame_history.append(
            self.processed_frames
        )

        self.motion_history.append(
            motion_area
        )

        self.prev_gray = gray

        # =====================================================
        # WRITE OUTPUT VIDEO
        # =====================================================

        if self.output_writer is not None:

            self.output_writer.write(
                output_frame
            )

        # =====================================================
        # DISPLAY FRAME
        # =====================================================

        self.display_frame(
            output_frame
        )

        # =====================================================
        # UPDATE GUI
        # =====================================================

        self.processed_label.config(
            text=str(
                self.processed_frames
            )
        )

        self.motion_label.config(
            text=str(
                self.motion_frames
            )
        )

        self.current_label.config(
            text=f"{displacement:.2f} px"
        )

        self.distance_label.config(
            text=f"{self.total_displacement:.2f} px"
        )

        self.maximum_label.config(
            text=f"{self.max_displacement:.2f} px"
        )

        # -----------------------------------------------------
        # Processing FPS
        # -----------------------------------------------------

        if self.start_time is not None:

            elapsed = (
                time.time()
                - self.start_time
            )

            if elapsed > 0:

                processing_fps = (
                    self.processed_frames
                    / elapsed
                )

            else:

                processing_fps = 0

            self.processing_fps_label.config(
                text=f"{processing_fps:.2f}"
            )

        # -----------------------------------------------------
        # Continue
        # -----------------------------------------------------

        delay = max(
            10,
            int(
                1000 /
                self.video_fps
            )
        )

        self.root.after(
            delay,
            self.process_frame
        )

    # =========================================================
    # DISPLAY FRAME
    # =========================================================

    def display_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        image.thumbnail(
            (800, 550)
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.config(
            image=photo,
            text=""
        )

        self.video_label.image = photo

        self.latest_frame = frame.copy()

    # =========================================================
    # MOTION GRAPH
    # =========================================================

    def show_graph(self):

        if len(
            self.motion_history
        ) < 2:

            messagebox.showwarning(
                "No Data",
                "Process the video first."
            )

            return

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            self.frame_history,
            self.motion_history,
            linewidth=2
        )

        plt.xlabel(
            "Frame Number"
        )

        plt.ylabel(
            "Motion Magnitude"
        )

        plt.title(
            "Motion Magnitude Across Video Frames"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        # Save graph for report
        graph_path = os.path.join(
            "screenshots",
            "motion_graph.png"
        )

        plt.savefig(
            graph_path,
            dpi=300
        )

        plt.show()

    # =========================================================
    # SAVE CURRENT FRAME
    # =========================================================

    def save_frame(self):

        if self.latest_frame is None:

            messagebox.showwarning(
                "Warning",
                "No processed frame is available."
            )

            return

        filename = os.path.join(
            "screenshots",
            "motion_output.png"
        )

        cv2.imwrite(
            filename,
            self.latest_frame
        )

        messagebox.showinfo(
            "Saved",
            "Processed frame saved successfully:\n\n"
            + filename
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.running = False
        self.paused = False

        if self.cap is not None:
            self.cap.release()

        if self.output_writer is not None:
            self.output_writer.release()

        self.cap = None
        self.output_writer = None

        self.prev_gray = None
        self.prev_center = None

        self.total_frames = 0
        self.processed_frames = 0
        self.motion_frames = 0

        self.total_displacement = 0.0
        self.max_displacement = 0.0

        self.motion_history = []
        self.frame_history = []

        self.start_time = None
        self.latest_frame = None
        self.video_path = None

        # Reset GUI
        self.video_label.config(
            image="",
            text="No video selected\n\nClick LOAD VIDEO"
        )

        self.video_value_label.config(
            text="0"
        )

        self.total_label.config(
            text="0"
        )

        self.processed_label.config(
            text="0"
        )

        self.motion_label.config(
            text="0"
        )

        self.current_label.config(
            text="0"
        )

        self.distance_label.config(
            text="0"
        )

        self.maximum_label.config(
            text="0"
        )

        self.processing_fps_label.config(
            text="0"
        )

        self.status.config(
            text="Status: Waiting",
            fg="#F7DC6F"
        )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = HumanMotionAnalysis(
        root
    )

    root.mainloop()
