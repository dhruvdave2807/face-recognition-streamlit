import os
import sys
import numpy as np
import face_recognition
from PIL import Image, ImageOps

# Minimum face size (in pixels) to consider for matching
# Lowered to handle your "not that small" image.
MIN_FACE_SIZE = 30


def load_image_correct_orientation(image_path):
    """
    Load an image with PIL, auto-rotate based on EXIF,
    convert to RGB (handles PNG with alpha),
    then convert to a NumPy array for face_recognition.
    """
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")  # <-- important line
    return np.array(pil_image)


def get_image_files(folder_path):
    """Return list of image file paths in a folder."""
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(exts)
    ]


def encode_face_from_image(image_path):
    """
    Load an image and return the first face encoding (if any).
    For selfie, we assume only one main face.
    """
    image = load_image_correct_orientation(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"[!] No face found in image: {image_path}")
        return None

    # Take the first face (for selfie that's fine)
    return encodings[0]


def find_person_in_photos(selfie_path, db_folder, tolerance=0.6):
    """
    selfie_path: path to the selfie image
    db_folder: folder containing many photos
    tolerance: smaller = stricter matching (default 0.6)
    """
    print("[*] Encoding selfie...")
    selfie_encoding = encode_face_from_image(selfie_path)
    if selfie_encoding is None:
        print("[x] Exiting: Couldn't detect a face in the selfie.")
        return []

    if not os.path.isdir(db_folder):
        print(f"[x] Folder not found: {db_folder}")
        return []

    image_paths = get_image_files(db_folder)
    print(f"[*] Found {len(image_paths)} images in database.\n")

    matched_images = []

    for img_path in image_paths:
        print(f"[*] Checking {img_path} ...")

        # Load image with orientation fixed
        image = load_image_correct_orientation(img_path)

        # Detect all faces in this image with some upsampling (helps with smaller faces)
        face_locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=2,  # increase if many small faces
            model="hog"                     # keep "hog" for speed; "cnn" needs extra model file
        )

        if len(face_locations) == 0:
            print("    - No faces found in this image.")
            continue

        # Get encodings for each detected face
        face_encodings = face_recognition.face_encodings(image, face_locations)

        min_distance = None
        found_match_here = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            w = right - left
            h = bottom - top

            # Skip very tiny faces (too far = unreliable embedding)
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                print(f"    - Face too small (w={w}, h={h}) → skipping")
                continue

            distance = face_recognition.face_distance([selfie_encoding], face_encoding)[0]

            if min_distance is None or distance < min_distance:
                min_distance = distance

            print(f"    - Face size: {w}x{h}, distance: {distance:.3f}")

            if distance <= tolerance:
                print(f"    -> MATCH ✅ (distance={distance:.3f})")
                matched_images.append(img_path)
                found_match_here = True
                break  # stop checking more faces in this image

        if min_distance is not None:
            print(f"    - Min distance (valid faces) in this image: {min_distance:.3f}")

        if not found_match_here:
            print("    -> No match in this image.")

    return matched_images


def main():
    # Usage:
    # python find_me.py <selfie_path> <images_folder> [tolerance]
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python find_me.py <selfie_path> <images_folder> [tolerance]")
        print("\nExample:")
        print("  python find_me.py selfie_path images_db 0.55")
        sys.exit(1)

    selfie_path = sys.argv[1]
    db_folder = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.6

    if not os.path.isfile(selfie_path):
        print(f"[x] Selfie not found: {selfie_path}")
        sys.exit(1)

    print(f"Selfie: {selfie_path}")
    print(f"Database folder: {db_folder}")
    print(f"Tolerance: {tolerance}")
    print(f"Min face size: {MIN_FACE_SIZE} px\n")

    results = find_person_in_photos(selfie_path, db_folder, tolerance)

    print("\n================ RESULT ================")
    if not results:
        print("No images found containing this person.")
    else:
        print("Person found in these images:")
        for r in results:
            print(" -", r)


if __name__ == "__main__":
    main()
