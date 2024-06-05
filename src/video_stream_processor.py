import cv2
from cuda_interface import edge_detection

class VideoStreamProcessor:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def process_stream(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = edge_detection(gray_frame)

            cv2.imshow('Processed Video Stream', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoStreamProcessor()
    processor.process_stream()
