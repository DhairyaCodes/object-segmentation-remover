# Generative AI Object Eraser  

An AI-powered image cleanup tool that detects objects, lets you choose which ones to remove, and uses advanced inpainting to restore the background — all in a few clicks.  
Powered by **YOLOv8**, **Segment Anything Model (SAM)**, and **LightX API**.  

---

## Demo

<img width="1919" height="944" alt="Screenshot 2025-08-12 131011" src="https://github.com/user-attachments/assets/4b85e422-9742-41cf-a7ac-92bd3520d0c6" />

<img width="1919" height="946" alt="Screenshot 2025-08-12 131045" src="https://github.com/user-attachments/assets/48074ad0-4427-4a39-b8f6-0e88ff10690d" />

---

## Features  
- **Automatic Object Detection** using YOLOv8  
- **Smart Segmentation** via SAM for precise mask generation  
- **Interactive Review UI** built with Streamlit  
- **AI-based Background Filling** using LightX inpainting  
- **FastAPI Backend** for seamless processing  
- **Downloadable Results** in high quality  

---

## Setup Instructions  

### 1. Install dependencies  
```bash
pip install -r requirements.txt
```
### 2. Configure environment variables
Create a .env file in the project root:
```bash
LIGHTX_API_KEY=your_lightx_api_key_here
```
You can get your API key from LightX Developer Portal.

### 3. Download required models
Place the following files in the root directory:
- sam_vit_b_01ec64.pth
- yolov8s.pt

---

## How to Run
Start the backend (FastAPI server):

```bash
uvicorn main:app --reload
```
Start the frontend (Streamlit UI):
```bash
streamlit run frontend.py
```

---

## Tech Stack
**Frontend:**
- Streamlit – Interactive UI for image upload, review, and download

**Backend:**
- FastAPI – API for object detection, segmentation, and cleanup
- YOLOv8 – Object detection
- Segment Anything Model (SAM) – Object segmentation
- LightX API – Inpainting & background fill

**Other Tools:**
- OpenCV – Image processing
- Pillow – Image manipulation
- NumPy – Array & image matrix handling
