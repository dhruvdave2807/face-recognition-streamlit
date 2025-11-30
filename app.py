import numpy as np
import streamlit as st
import face_recognition
from PIL import Image, ImageOps

# --- CONFIGS ---
MIN_FACE_SIZE = 30   # min width/height (px) to consider a face
DEFAULT_TOLERANCE = 0.5


# ---------- HELPER FUNCTIONS ----------

def load_image_correct_orientation_file(file) -> np.ndarray:
    """
    Load an uploaded file with PIL, fix EXIF orientation,
    convert to RGB, and return as NumPy array.
    """
    pil_image = Image.open(file)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def encode_face_from_image_array(image_array: np.ndarray):
    """
    Given an RGB numpy image, return the first face encoding if found,
    otherwise None.
    """
    encodings = face_recognition.face_encodings(image_array)

    if len(encodings) == 0:
        return None

    return encodings[0]


def find_matches(selfie_encoding, photos_files, tolerance=DEFAULT_TOLERANCE):
    """
    Loop over uploaded photos, detect faces, and return
    list of matches: [(file_name, best_distance, image_array), ...]
    """
    matches = []

    for file in photos_files:
        try:
            image_array = load_image_correct_orientation_file(file)
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")
            continue

        # Detect faces (upsample for smaller faces)
        face_locations = face_recognition.face_locations(
            image_array,
            number_of_times_to_upsample=2,
            model="hog"
        )

        if len(face_locations) == 0:
            st.write(f"🔍 No faces detected in **{file.name}**.")
            continue

        face_encodings = face_recognition.face_encodings(
            image_array,
            face_locations
        )

        best_distance = None
        is_match = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            w = right - left
            h = bottom - top

            # Skip very tiny faces
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            distance = face_recognition.face_distance(
                [selfie_encoding],
                face_encoding
            )[0]

            if best_distance is None or distance < best_distance:
                best_distance = distance

            if distance <= tolerance:
                is_match = True

        if is_match and best_distance is not None:
            matches.append((file.name, best_distance, image_array))

    return matches


# ---------- STREAMLIT APP UI ----------

st.set_page_config(page_title="Find Me In Photos", page_icon="🧠")

st.title("🧑‍💻 Find My Face In Photos")
st.write(
    "Upload a **selfie** and a bunch of **photos**. "
    "The app will highlight which photos contain your face."
)

st.sidebar.header("Settings")
tolerance = st.sidebar.slider(
    "Match tolerance (lower = stricter, fewer false positives)",
    min_value=0.3,
    max_value=0.7,
    value=DEFAULT_TOLERANCE,
    step=0.01,
)
st.sidebar.write(f"Current tolerance: **{tolerance:.2f}**")
st.sidebar.write(f"Minimum face size: **{MIN_FACE_SIZE}px**")


# --- File uploaders ---

selfie_file = st.file_uploader(
    "Step 1: Upload your selfie",
    type=["jpg", "jpeg", "png"],
    key="selfie",
)

photos_files = st.file_uploader(
    "Step 2: Upload photos to search (you can select multiple)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="photos",
)

if st.button("🔎 Find me in photos"):
    if selfie_file is None:
        st.error("Please upload a selfie first.")
    elif not photos_files:
        st.error("Please upload at least one photo to search in.")
    else:
        # Process selfie
        with st.spinner("Encoding your selfie..."):
            selfie_image_array = load_image_correct_orientation_file(selfie_file)
            selfie_encoding = encode_face_from_image_array(selfie_image_array)

        if selfie_encoding is None:
            st.error("No face detected in the selfie. Try another selfie (clear, front-facing).")
        else:
            st.success("Selfie encoded successfully ✅")
            st.write("---")

            # Search in uploaded photos
            with st.spinner("Searching for your face in uploaded photos..."):
                matches = find_matches(selfie_encoding, photos_files, tolerance=tolerance)

            st.write("## Results")

            if not matches:
                st.warning("No matching faces found in the uploaded photos at this tolerance.")
            else:
                st.success(f"Found your face in **{len(matches)}** photo(s):")

                for file_name, distance, image_array in matches:
                    st.write(f"**{file_name}** — distance: `{distance:.3f}`")
                    st.image(image_array, use_column_width=True)
                    st.write("---")
