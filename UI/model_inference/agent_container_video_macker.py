import os
import cv2
import torch
from ultralytics import YOLO


# Method 1: Detect container back and generate output video
def container_back_video_gen(video_path: str) -> str:
    model_path = "/home/ubuntu/navaoceanlog/UI/models/container_back_identifier.pt"
    output_path = video_path.replace(".mp4", "_container_back.mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    
    print(f"[INFO] Process PID: {os.getpid()} - Use this PID for GPU monitoring")
    print(f"[INFO] Device: {device}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Running container back detection on: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, device=device)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Saved container back output to: {output_path}")
    return output_path


# Method 2: Detect labels on container back and generate new video
def container_back_label_gen(video_path: str) -> str:
    model_path = "/home/ubuntu/navaoceanlog/UI/models/container_back_labeled.pt"
    output_path = video_path.replace(".mp4", "_labelled.mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    
    print(f"[INFO] Process PID: {os.getpid()} - Use this PID for GPU monitoring")
    print(f"[INFO] Device: {device}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Running label detection on: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, device=device)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Saved label output to: {output_path}")
    return output_path


# Method 3: Combine both steps
def generate_final_output(input_video_path: str) -> str:
    intermediate_video = container_back_video_gen(input_video_path)
    final_output_video = container_back_label_gen(intermediate_video)
    return final_output_video


# Main driver
if __name__ == "__main__":
    input_video = "/home/ubuntu/navaoceanlog/UI/uploaded_video_files/short_2.mp4"  # Replace with your input video path
    final_output = generate_final_output(input_video)
    print(f"[DONE] Final video generated at: {final_output}")
