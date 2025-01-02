import os
os.system("pip install -U ultralytics open_clip_torch numpy opencv-python gradio")

import spaces
import gradio as gr
from ultralytics import YOLO
import torch
import open_clip
import cv2
import numpy as np
from PIL import Image

yolo_model = YOLO("yolov8s-worldv2.pt")
yolo_model.set_classes(["electric-bike", "electric-bicycle", "e-bike", "bicycle"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("RN50-quickgelu", pretrained="openai", device=device)
model.eval()
tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

classes = ["an e-bike", "a bicycle", "an electric-bike", "an electric-bicycle"]
pretty_classes = ["e-bike", "bicycle", "e-bike", "e-bike"]

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def process_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    font_scale = max(width, height) / 1000
    thickness = int(max(width, height) / 300)
    results = yolo_model(img_rgb, iou=0.5, conf=0.25)
    boxes, confidences, class_ids = [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)
            class_ids.append(class_id)

    to_remove = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > 0.5:
                if confidences[i] > confidences[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    boxes = [box for idx, box in enumerate(boxes) if idx not in to_remove]
    confidences = [conf for idx, conf in enumerate(confidences) if idx not in to_remove]
    class_ids = [cls for idx, cls in enumerate(class_ids) if idx not in to_remove]

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        cropped_img = img_rgb[y1:y2, x1:x2]
        image_input = preprocess(Image.fromarray(cropped_img)).unsqueeze(0).to(device)
        text_inputs = tokenizer(classes).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        predicted_class = pretty_classes[probs.argmax()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"{predicted_class} {probs.max():.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    return frame

@spaces.GPU
def detect_objects_image(image):
    image_np = np.array(image)
    processed_image = process_frame(image_np)
    return Image.fromarray(processed_image)

@spaces.GPU
def detect_objects_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        frames.append(processed_frame)
    cap.release()
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter("/tmp/annotated_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return "/tmp/annotated_video.mp4"

global_title = "DeepSense.ai"

file_upload_demo = gr.Interface(
    fn=detect_objects_image,
    inputs=gr.Image(type="pil", label="Input"),
    outputs=gr.Image(type="pil", label="Output"),
    title="Image Upload - Bicycle and E-Bike Detection Model",
    description=(
        "<span style='font-size: 17px;'>"
        "- <b>Step 1</b>: Upload an image, video or live video in the upload box below.<br>"
        "- <b>Step 2</b>: Press <code>Submit</code> and the detection model will label the image or video in the output box."
        "</span>"
    ),
    flagging_options=["Incorrect"],
    allow_flagging="manual"
)

video_file_demo = gr.Interface(
    fn=detect_objects_video,
    inputs=gr.Video(label="Input"),
    outputs=gr.Video(label="Output"),
    title="Video Upload - Bicycle and E-Bike Detection Model",
    description=(
        "<span style='font-size: 17px;'>"
        "- <b>Step 1</b>: Upload an image, video or live video in the upload box below.<br>"
        "- <b>Step 2</b>: Press <code>Submit</code> and the detection model will label the image or video in the output box."
        "</span>"
    ),
    flagging_options=["Incorrect"],
    allow_flagging="manual"
)

live_video_demo = gr.Interface(
    fn=detect_objects_image,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Input"),
    outputs=gr.Image(type="pil", label="Output"),
    live=True,
    title="Live Video - Bicycle and E-Bike Detection Model",
    description=(
        "<span style='font-size: 17px;'>"
        "- <b>Step 1</b>: Upload an image, video or live video in the upload box below.<br>"
        "- <b>Step 2</b>: Press <code>Submit</code> and the detection model will label the image or video in the output box."
        "</span>"
    ),
    flagging_options=["Incorrect"],
    allow_flagging="manual"
)

with gr.Blocks(title=global_title) as demo:
    gr.Markdown("<h1 style='text-align: center;'>DeepSense.ai</h1>")
    gr.TabbedInterface(
        [file_upload_demo, video_file_demo, live_video_demo],
        ["Image Upload", "Video Upload", "Live Video"],
    )
    gr.Markdown(
        "<div style='text-align: center; font-size: 15px; margin-top: 20px;'>"
        "Please press the <code>Flag</code> button under the output box if the label is incorrect, this will help the model with reinforcement learning."
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(share=True)
