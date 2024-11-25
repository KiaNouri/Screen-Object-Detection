import mss
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch


class ScreenObjectDetector:
    def __init__(self, confidence=0.5):
        # Initialize the screen capture
        self.sct = mss.mss()

        # Load YOLO model
        self.model = YOLO('yolo11x.pt')
        self.confidence = confidence

        # Get color list for visualization
        self.colors = self.generate_colors()

    def generate_colors(self):
        # Generate random colors for each class
        np.random.seed(42)
        return np.random.randint(0, 255, size=(80, 3)).tolist()

    def capture_screen(self):
        # Capture monitor and process sceenshots
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def process_frame(self, frame):
        # perform detection and process results
        results = self.model(frame, conf=self.confidence)

        annotated_frame = frame.copy()
        for result in results:
            # draw bounding boxes
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                color = self.colors[cls]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # handle label
                label = f'{class_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1,
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return annotated_frame

    def start_detection(self, save_video=True, output_path='./screen_recording.mp4'):
        try:
            # create awindow
            cv2.namedWindow('Screen Object Detection', cv2.WINDOW_NORMAL)

            # Initiate video writer if saving is enabled
            video_writer = None
            if save_video:
                screen = self.capture_screen()
                height, width = screen.shape[:2]
                video_writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5,
                    (width, height),
                )

            # fps counter
            fps_start_time = time.time()
            fps_counter = 0
            fps = 0
            while True:
                screen = self.capture_screen()
                processed_screen = self.process_frame(screen)

                fps_counter += 1
                if (time.time() - fps_start_time) > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                cv2.putText(
                    processed_screen,
                    f'FPS: {fps}',
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Show results
                cv2.imshow('Screen Object Detection', processed_screen)

                if video_writer is not None:
                    video_writer.write(processed_screen)

                # Break if pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()


def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_video_path = './screen_recordingv2.mp4'

    # Create and start detector
    detector = ScreenObjectDetector(confidence=0.5)
    detector.start_detection(save_video=True, output_path=output_video_path)


if __name__ == "__main__":
    main()
