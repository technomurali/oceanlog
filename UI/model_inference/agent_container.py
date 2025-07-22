"""
Final version combines:
  • GPU/CPU auto-detection (prints active device)
  • Robust EOF handling (guards against last-frame replay)
  • Detailed debug logging (progress, saves, summary)
  • Execution-time measurement

Save as detect_container_backs.py and run directly or import.
"""

import os
import time
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import os
from paddleocr import PaddleOCR
import json


import logging
from typing import List, Dict, Any


import os
import cv2
import torch
from ultralytics import YOLO

def detect_and_crop_container_backs(
    model_path: str,
    video_path: str,
    output_dir: str,
    label_of_interest: str = "container_back",
    confidence_threshold: float = 0.80,
    progress_every_n: int = 50  # heartbeat frequency
) -> None:
    """
    Detect `label_of_interest` objects in `video_path` using YOLOv8,
    crop detections above `confidence_threshold` with 10px margin,
    and save to `output_dir`. Works on CPU or GPU.
    """
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    model = YOLO(model_path).to(device_str)

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INIT] Running on GPU: {gpu_name}")
    else:
        print("[INIT] Running on CPU")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] total_frames={total_frames}, fps={fps}")

    frame_count = 0
    saved_count = 0
    consecutive_same_frame = 0
    last_frame_idx = -1

    # ------------------------------------------------------------------
    # Frame processing loop
    # ------------------------------------------------------------------
    while True:
        ret, frame = cap.read()

        # EOF or read failure
        if not ret or frame is None:
            print(f"[EOF] cap.read() returned ret={ret}, frame is None? {frame is None}")
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # one-based
        if frame_idx == last_frame_idx:
            consecutive_same_frame += 1
        else:
            consecutive_same_frame = 0
        last_frame_idx = frame_idx

        # Guard: stop if decoder keeps replaying the last frame
        if consecutive_same_frame > 5:
            print("[GUARD] same frame returned >5 times breaking loop")
            break

        frame_count += 1
        if frame_count > total_frames + 5:
            print("[GUARD] frame_count exceeded expected total breaking loop")
            break

        if frame_count % progress_every_n == 0:
            print(f"[PROGRESS] frame {frame_count}/{total_frames}")

        # Inference
        results = model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False,
            device=device_str,
        )[0]

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))

            if label == label_of_interest and conf >= confidence_threshold:
                height, width = frame.shape[:2]
                margin = 20

                # Original box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Expand with margin, clamp to image bounds
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)

                crop = frame[y1:y2, x1:x2]

                cv2.putText(
                    crop,
                    f"{conf * 100:.1f}%",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                filename = f"{label}_{frame_count}_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), crop)
                saved_count += 1
                print(f"[SAVE] {filename} (conf {conf:.3f})")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # flush any pending HighGUI events

    print("[EXIT] done.")




def clean_folder_except_last_n(folder_path: str, last_no_files: int = 1):
    # List only regular files
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]

    # Nothing to do if there aren't more files than we want to keep
    if len(files) <= last_no_files:
        print(f"Found {len(files)} file(s). No deletion needed.")
        return

    file_tuples = []
    for fname in files:
        parts = fname.rsplit('_', 2)  
        # e.g. ['container', 'back', '114_0'] if no extension
        # or ['container', 'back', '114', '0.jpg'] depending on your naming
        if len(parts) >= 2:
            # take the part before the final underscore
            seq_str = parts[-2]
            try:
                seq_num = int(seq_str)
                file_tuples.append((seq_num, fname))
            except ValueError:
                print(f"Skipping (bad sequence) → {fname}")
        else:
            print(f"Skipping (unexpected format) → {fname}")

    if len(file_tuples) <= last_no_files:
        print(f"Only found {len(file_tuples)} numbered file(s). No deletion needed.")
        return

    # sort by the extracted sequence number
    file_tuples.sort(key=lambda x: x[0])

    # decide which to keep
    to_keep = set(f for _, f in file_tuples[-last_no_files:])

    # delete the rest
    for _, fname in file_tuples:
        full = os.path.join(folder_path, fname)
        if fname in to_keep:
            print(f"Kept:    {fname}")
        else:
            os.remove(full)
            print(f"Deleted: {fname}")


# Example usage:
# clean_folder_except_last_n("D:/data/container_images", last_no_files=2)


def detect_and_crop_labels(model_path: str, image_folder: str, output_root: str = None):
    """
    Loads a YOLO model and performs label detection on the first image found in `image_folder`.
    Crops the detected labels and saves them to a subfolder based on the image name.
    """
    # Locate first image file in the given folder
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_file = next((f for f in os.listdir(image_folder) if Path(f).suffix.lower() in supported_exts), None)
    if not image_file:
        raise FileNotFoundError(f"No image file found in {image_folder}")

    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    # Load YOLO model
    model = YOLO(model_path)

    # Run inference using GPU (device=0)
    results = model.predict(source=image, save=False, device=0)[0]

    # Setup output directory
    image_name = Path(image_file).stem
    if output_root is None:
        output_root = image_folder
    image_output_dir = os.path.join(output_root, f"labels")
    os.makedirs(image_output_dir, exist_ok=True)

    # Crop and save each detected box
    for i, box in enumerate(results.boxes):
        conf = box.conf[0].item()
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cropped = image[y1:y2, x1:x2]
        output_file = os.path.join(image_output_dir, f"{label}.jpg")
        cv2.imwrite(output_file, cropped)

    print(f"Cropped {len(results.boxes)} objects and saved to: {image_output_dir}")




def run_paddle_ocr_on_folder(image_folder: str, output_filename: str = 'ocr_results.txt'):
    """
    Runs OCR on all images in a folder using PaddleOCR and writes results to a text file.

    Args:
        image_folder (str): Folder containing images (.jpg/.jpeg/.png).
        output_filename (str): Output file name for OCR results.
    """

    output_file = os.path.join(image_folder, output_filename)

    # Initialize OCR engine — no deprecated arguments
    ocr = PaddleOCR(lang='en')  # Automatically uses GPU if paddlepaddle-gpu is installed

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for image_name in os.listdir(image_folder):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_name)
                result = ocr.ocr(image_path, cls=True)

                lines = []
                for line in result[0]:
                    text = line[1][0]
                    lines.append(text)

                extracted_text = " | ".join(lines)
                filename_without_ext = os.path.splitext(image_name)[0]
                f_out.write(f"{filename_without_ext}: {extracted_text}\n")

    print(f"OCR processing completed. Results saved to {output_file}")


def run_paddle_ocr_to_json(
    image_folder: str,
    output_filename: str = "ocr_results.json",
    *,
    lang: str = "en",
    logger: logging.Logger | None = None,
    ocr_kwargs: Dict[str, Any] | None = None,
) -> str:
    """
    Runs PaddleOCR on image files in `image_folder` and writes consolidated
    results to `<image_folder>/<output_filename>` in JSON format.

    Args
    ----
    image_folder : str
        Directory containing images (`.jpg`, `.jpeg`, `.png`).
    output_filename : str, default "ocr_results.json"
        Name of the JSON file to create in `image_folder`.
    lang : str, default "en"
        Language model to load in PaddleOCR.
    logger : logging.Logger | None
        Optional logger.  If None, a basic logger is configured automatically.
    ocr_kwargs : dict | None
        Extra keyword arguments forwarded to `PaddleOCR(**ocr_kwargs)`.

    Returns
    -------
    str
        Absolute path to the generated JSON file.

    Raises
    ------
    FileNotFoundError
        If `image_folder` does not exist or contains no image files.
    RuntimeError
        If every image fails OCR (e.g., all corrupt or unreadable).
    """
    # ------------------------------------------------------------------ setup
    if logger is None:  # create a default logger if the caller did not supply
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logger = logging.getLogger(__name__)

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    # discover images (case-insensitive)
    def _is_image(fname: str) -> bool:
        return fname.lower().endswith((".jpg", ".jpeg", ".png"))

    images: List[str] = sorted(f for f in os.listdir(image_folder) if _is_image(f))
    if not images:
        raise FileNotFoundError(f"No images found in folder: {image_folder}")

    # build full output path early so that write-errors are detected up-front
    output_file = os.path.abspath(os.path.join(image_folder, output_filename))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Instantiate PaddleOCR once; GPU/CPU handled internally
    ocr_params = ocr_kwargs or {}
    logger.info("Loading PaddleOCR (lang=%s)…", lang)
    ocr_engine = PaddleOCR(lang=lang, **ocr_params)  # may take a few seconds

    results_list: List[Dict[str, Any]] = []
    failures: List[str] = []

    # ----------------------------------------------------------- per-image OCR
    for idx, image_name in enumerate(images, 1):
        image_path = os.path.join(image_folder, image_name)
        labelname = os.path.splitext(image_name)[0]  # remove extension

        try:
            result = ocr_engine.ocr(image_path, cls=True)
            if (
                not result                     # Paddle returned None/empty
                or not isinstance(result, list)
                or not isinstance(result[0], list)
            ):
                logger.warning(
                    "No text detected in '%s'; skipping.", image_name
                )
                continue

            lines: List[str] = [line[1][0] for line in result[0] if line and line[1]]
            results_list.append({"labelname": labelname, "text": lines})
            logger.info("(%d/%d) OK: %s", idx, len(images), image_name)

        except Exception as err:  # catch *all* errors per-file
            failures.append(image_name)
            logger.exception("(%d/%d) ERROR processing '%s': %s",
                             idx, len(images), image_name, err)

    # -------------------------------------------------------------- post-write
    if not results_list:
        raise RuntimeError(
            "OCR finished but produced zero successful results; "
            "see earlier log messages for details."
        )

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(results_list, fp, ensure_ascii=False, indent=2)

    logger.info(
        "OCR complete → %s (success: %d, failed: %d)",
        output_file, len(results_list), len(failures)
    )
    return output_file

def run_paddle_ocr_to_json_old(image_folder: str, output_filename: str = 'ocr_results.json'):
    """
    Runs OCR on all images in the folder and saves results as JSON.
    
    Each entry in the JSON will have:
        {
            "labelname": "chassis_license_plate",
            "text": ["ABC123", "OtherLabel", ...]
        }
    """
    output_file = os.path.join(image_folder, output_filename)
    ocr = PaddleOCR(lang='en')

    results_list = []

    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_name)
            result = ocr.ocr(image_path, cls=True)

            lines = [line[1][0] for line in result[0]]

            labelname = os.path.splitext(image_name)[0]  # remove extension
            results_list.append({
                "labelname": labelname,
                "text": lines
            })

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results_list, f_out, ensure_ascii=False, indent=2)

    print(f"OCR JSON results saved to {output_file}")


def generate_output_path(input_path: str, label: str, output_base_path: str) -> str:
    """
    Generates a dynamic output path by appending the input video's base name (without extension)
    and a label to the output base directory.

    Args:
        input_path (str): Full path to the input video file (e.g., /path/to/video.mp4)
        label (str): Label to append (e.g., "container_back")
        output_base_path (str): Base output directory (e.g., /output/path)

    Returns:
        str: Final output directory path (e.g., /output/path/v4_container_back)
    """
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Create new folder name
    new_folder_name = f"{base_name}_{label}"

    # Join with output base path
    final_output_path = os.path.join(output_base_path, new_folder_name)

    return final_output_path



# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

# video_path="/home/ubuntu/navaoceanlog/UI/uploaded_video_files/v4.mp4",
def run_all(input_video_path_setup: str,output_dair_path:str):
        
#    Model path 
    model_container_back_path = "/home/ubuntu/navaoceanlog/UI/models/container_back_identifier.pt"
    model_container_back_labele = "/home/ubuntu/navaoceanlog/UI/models/container_back_labeled.pt"
    
    output_dir=generate_output_path(input_video_path_setup,"container_back",output_dair_path)
    detect_and_crop_container_backs(
        model_path=model_container_back_path,
        video_path=input_video_path_setup,
        output_dir=output_dir,
        label_of_interest="container_back",
        confidence_threshold=0.953,
        progress_every_n=25,
    )
    clean_folder_except_last_n(output_dir)


    image_folder =output_dir
    detect_and_crop_labels(model_container_back_labele, image_folder)
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PaddleOCR is not installed in the current environment. "
            "Install it with:  pip install paddleocr"
    ) from exc
    
    run_paddle_ocr_to_json(f'{output_dir}/labels')


if __name__ == "__main__":
    run_all("/home/ubuntu/navaoceanlog/UI/long/v6.mp4","/home/ubuntu/navaoceanlog/UI/video_outputs")