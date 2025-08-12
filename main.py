# --- Imports ---
import io
import os
import uuid
from typing import List, Dict
import base64
import time
from dotenv import load_dotenv

import cv2
import numpy as np
import torch
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from starlette.middleware.cors import CORSMiddleware

# --- Global Variables & Model Loading ---
SESSION_STORE: Dict[str, Dict] = {}

print("Loading models, this might take a moment...")

load_dotenv()
LIGHTX_API_KEY = os.getenv('LIGHTX_API_KEY')
if not LIGHTX_API_KEY:
    print("WARNING: LIGHTX_API_KEY not available")

DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# --- YOLO Model ---
try:
    YOLO_MODEL = YOLO("yolov8s.pt")
    YOLO_MODEL.to(DEVICE)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# --- SAM (Segment Anything Model) ---
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(SAM_CHECKPOINT_PATH):
    print("Downloading SAM model...")
    r = requests.get(SAM_MODEL_URL, stream=True)
    r.raise_for_status()
    with open(SAM_CHECKPOINT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("SAM model downloaded successfully.")

try:
    SAM_MODEL = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT_PATH)
    SAM_MODEL.to(device=DEVICE)
    SAM_PREDICTOR = SamPredictor(SAM_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to load SAM model: {e}")

print("Models loaded successfully.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Generative AI Object Eraser Backend",
    description="An API to detect objects, generate masks, and use LightX for inpainting.",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def image_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    """Converts a NumPy array image to a byte string."""
    is_success, buffer = cv2.imencode(f".{format.lower()}", img)
    if not is_success:
        raise ValueError(f"Failed to encode image to {format}")
    return buffer.tobytes()

def create_visual_mask(mask: np.ndarray, random_color: bool = True) -> np.ndarray:
    """Creates a colored, semi-transparent visual mask for the frontend."""
    visual_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    color = np.random.randint(0, 255, size=3) if random_color else np.array([255, 0, 0])
    visual_mask[mask > 0, :3] = color
    visual_mask[mask > 0, 3] = 150
    return visual_mask

def upload_to_lightx(image_bytes: bytes, content_type: str, api_key: str) -> str:
    """Handles the two-step upload process for LightX and returns the final image URL."""
    get_url_endpoint = "https://api.lightxeditor.com/external/api/v2/uploadImageUrl"
    headers = {"Content-Type": "application/json", "x-api-key": api_key}
    payload = {"uploadType": "imageUrl", "size": len(image_bytes), "contentType": content_type}
    
    response = requests.post(get_url_endpoint, headers=headers, json=payload)
    response.raise_for_status()
    upload_data = response.json()

    if upload_data.get("statusCode") != 2000:
        raise HTTPException(status_code=500, detail=f"LightX failed to provide upload URL: {upload_data.get('message')}")

    upload_url = upload_data["body"]["uploadImage"]
    final_image_url = upload_data["body"]["imageUrl"]

    put_headers = {"Content-Type": content_type}
    put_response = requests.put(upload_url, headers=put_headers, data=image_bytes)
    put_response.raise_for_status()

    return final_image_url


# --- API Endpoints ---

@app.get("/", summary="Root endpoint to check if the server is running.")
def read_root():
    return {"status": "ok", "message": "Object Eraser API with LightX is running."}


@app.post("/process-image", summary="Detect objects and generate masks.")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    image_np = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Running YOLO detection...")
    yolo_results = YOLO_MODEL(image_rgb, conf=0.4)
    
    if len(yolo_results[0].boxes) == 0:
        raise HTTPException(status_code=404, detail="No objects were detected in the image.")

    print(f"Found {len(yolo_results[0].boxes)} potential objects. Generating masks with SAM...")
    SAM_PREDICTOR.set_image(image_rgb)
    boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
    transformed_boxes = SAM_PREDICTOR.transform.apply_boxes_torch(torch.from_numpy(boxes_xyxy).to(DEVICE), image_rgb.shape[:2])
    
    masks, _, _ = SAM_PREDICTOR.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
    masks = masks.cpu().numpy().squeeze(1)

    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = {"original_image": contents, "all_masks": masks}

    visual_masks_bytes = [image_to_bytes(create_visual_mask(m), "PNG") for m in masks]
    print(f"Session {session_id} created with {len(masks)} masks.")

    original_image_b64 = base64.b64encode(contents).decode('utf-8')
    visual_masks_b64 = [base64.b64encode(m).decode('utf-8') for m in visual_masks_bytes]

    return JSONResponse(content={
        "session_id": session_id,
        "original_image_b64": original_image_b64,
        "masks_b64": visual_masks_b64,
        "message": f"Successfully processed image. Found {len(masks)} objects."
    })


@app.post("/cleanup-image", summary="Remove selected objects from the image using LightX.")
async def cleanup_image(session_id: str = Body(...), selected_indices: List[int] = Body(...)):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_data = SESSION_STORE[session_id]
    original_image_bytes = session_data["original_image"]
    all_masks = session_data["all_masks"]

    if not selected_indices:
        del SESSION_STORE[session_id]
        print(f"No objects selected. Returning original image for session {session_id}.")
        return StreamingResponse(io.BytesIO(original_image_bytes), media_type="image/png")

    if max(selected_indices) >= len(all_masks):
        raise HTTPException(status_code=400, detail="Invalid selection of objects to remove.")

    print(f"Combining {len(selected_indices)} masks for session {session_id}...")
    final_mask = np.zeros_like(all_masks[0], dtype=np.uint8)
    for index in selected_indices:
        final_mask[all_masks[index]] = 255
    
    final_mask_bytes = image_to_bytes(final_mask, "PNG")

    if not LIGHTX_API_KEY:
        raise HTTPException(status_code=500, detail="LIGHTX_API_KEY is not configured on the server.")

    try:
        print("Uploading original image to LightX...")
        image_url = upload_to_lightx(original_image_bytes, "image/png", LIGHTX_API_KEY)
        
        print("Uploading mask to LightX...")
        mask_url = upload_to_lightx(final_mask_bytes, "image/png", LIGHTX_API_KEY)

        print("Submitting cleanup job to LightX...")
        cleanup_endpoint = "https://api.lightxeditor.com/external/api/v1/cleanup-picture"
        headers = {"Content-Type": "application/json", "x-api-key": LIGHTX_API_KEY}
        payload = {"imageUrl": image_url, "maskedImageUrl": mask_url}
        
        cleanup_response = requests.post(cleanup_endpoint, headers=headers, json=payload)
        cleanup_response.raise_for_status()
        cleanup_data = cleanup_response.json()
        
        if cleanup_data.get("statusCode") != 2000:
            raise HTTPException(status_code=500, detail=f"LightX failed to start cleanup job: {cleanup_data.get('message')}")

        order_id = cleanup_data["body"]["orderId"]
        print(f"Cleanup job started with orderId: {order_id}")

        status_endpoint = "https://api.lightxeditor.com/external/api/v1/order-status"
        status_payload = {"orderId": order_id}
        max_retries = 5
        
        for i in range(max_retries):
            print(f"Checking status... Attempt {i + 1}/{max_retries}")
            time.sleep(3)
            
            status_response = requests.post(status_endpoint, headers=headers, json=status_payload)
            status_response.raise_for_status()
            status_data = status_response.json()["body"]
            
            if status_data["status"] == "active":
                print("Job complete! Fetching final image.")
                output_url = status_data["output"]
                final_image_response = requests.get(output_url, stream=True)
                final_image_response.raise_for_status()
                
                del SESSION_STORE[session_id]
                print(f"Session {session_id} cleaned up.")
                
                return StreamingResponse(final_image_response.iter_content(chunk_size=8192), media_type=final_image_response.headers['Content-Type'])
            
            elif status_data["status"] == "failed":
                raise HTTPException(status_code=500, detail="LightX cleanup job failed.")
        
        raise HTTPException(status_code=504, detail="LightX job timed out. The result was not ready in time.")

    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if e.response is not None:
            try:
                error_body = e.response.json()
                error_detail = error_body.get('error', e.response.text)
            except:
                error_detail = e.response.text
        print(f"An error occurred during the LightX API call: {error_detail}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing with LightX: {error_detail}")
