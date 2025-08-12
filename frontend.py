# frontend.py
import streamlit as st
import requests
import io
from PIL import Image
import base64

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Helper Functions ---
def reset_session_state():
    """Clears the session state to start over."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    if 'review_images' not in st.session_state:
        st.session_state.review_images = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'current_mask_index' not in st.session_state:
        st.session_state.current_mask_index = 0
    if 'indices_to_remove' not in st.session_state:
        st.session_state.indices_to_remove = []
    if 'final_image' not in st.session_state:
        st.session_state.final_image = None


# --- Streamlit UI Config ---
st.set_page_config(layout="wide")
st.title("🖼️ Generative AI Object Eraser")
st.markdown("Powered by YOLO, SAM, and LightX. Upload an image, choose which objects to remove, and let the AI do the rest.")
st.markdown(
    """
    <style>
    .stApp [data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding-top: 0.4rem; /* Adjust this value as needed, 0rem removes most padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initialize state
initialize_session_state()

# --- Main App Logic ---

# Step 1: Image Upload
if not st.session_state.processing_started:
    uploaded_file = st.file_uploader(
        "Choose an image to begin...", 
        type=["png", "jpg", "jpeg"], 
        key="file_uploader"
    )

    if uploaded_file is not None:
        st.session_state.processing_started = True
        with st.spinner("Processing image and preparing review... This may take a moment."):
            try:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/process-image", files=files, timeout=120)
                response.raise_for_status()

                data = response.json()
                st.session_state.session_id = data["session_id"]

                original_image_bytes = base64.b64decode(data["original_image_b64"])
                original_image = Image.open(io.BytesIO(original_image_bytes))

                review_images = []
                for mask_b64 in data["masks_b64"]:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_image = Image.open(io.BytesIO(mask_bytes))
                    display_image = original_image.copy().convert("RGBA")
                    display_image.paste(mask_image, (0, 0), mask_image)
                    review_images.append(display_image)
                
                st.session_state.review_images = review_images
                st.rerun()

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
                try:
                    st.error(f"Backend response: {response.json().get('detail', response.text)}")
                except:
                    st.error(f"Backend response: {response.text}")
                reset_session_state()


# Step 2: Mask Review Slideshow
elif st.session_state.processing_started and not st.session_state.final_image:
    st.header("Review Objects to Remove")
    
    total_masks = len(st.session_state.review_images)
    current_index = st.session_state.current_mask_index

    if total_masks == 0:
        st.warning("No objects were detected in the image. Please try another one.")
        if st.button("Start Over"):
            reset_session_state()
            st.rerun()
    elif current_index < total_masks:
        st.subheader(f"Object {current_index + 1} of {total_masks}")

        _, image_col, _ = st.columns([1, 1, 1])
        with image_col:
            st.image(st.session_state.review_images[current_index], caption="Do you want to remove the highlighted object?", use_container_width=True)

        _, btn_col1, btn_col2, _ = st.columns([1.5, 1, 1, 1.5])
        with btn_col1:
            if st.button("✅ Keep", key=f"keep_{current_index}", use_container_width=True):
                st.session_state.current_mask_index += 1
                st.rerun()
        with btn_col2:
            if st.button("❌ Remove", key=f"remove_{current_index}", use_container_width=True):
                st.session_state.indices_to_remove.append(current_index)
                st.session_state.current_mask_index += 1
                st.rerun()
    else:
        st.success("All objects reviewed!")
        if st.session_state.indices_to_remove:
            st.write(f"You have selected {len(st.session_state.indices_to_remove)} object(s) for removal.")
        else:
            st.write("You have not selected any objects for removal.")

        if st.button("✨ Generate Final Image", type="primary"):
            with st.spinner("AI is cleaning the image... This can take up to a minute."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "selected_indices": st.session_state.indices_to_remove
                    }
                    response = requests.post(f"{BACKEND_URL}/cleanup-image", json=payload, timeout=180)
                    response.raise_for_status()

                    final_image_bytes = response.content
                    st.session_state.final_image = final_image_bytes
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"Error during final processing: {e}")
                    try:
                        st.error(f"Backend response: {response.json().get('detail', response.text)}")
                    except:
                        st.error(f"Backend response: {response.text}")


# Step 3: Display and Download Final Image
elif st.session_state.final_image:
    st.header("✨ Your Cleaned Image is Ready!")
    
    _, image_col, _ = st.columns([1, 1, 1])
    with image_col:
        final_image = Image.open(io.BytesIO(st.session_state.final_image))
        st.image(final_image, caption="Final Result", use_container_width=True)

        st.download_button(
            label="📥 Download Image",
            data=st.session_state.final_image,
            file_name="cleaned_image.png",
            mime="image/png",
            use_container_width=True
        )

    if st.button("Start Over"):
        reset_session_state()
        st.rerun()
