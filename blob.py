
"""
EasyBlobTracker - Audio-reactive blob visualization tool
Generates blobs that react to audio volume and motion detection
"""

import cv2
import numpy as np
import random
import time
from PIL import ImageFont, ImageDraw, Image
import librosa

# ============================================================================
# CONFIG & STYLIZATION PARAMETERS
# ============================================================================

# File paths
FONT_PATH = r"static\ibmreg.ttf"
VIDEO_PATH = r"static\video.mp4"
AUDIO_PATH = r"static\song.wav"
OUTPUT_NAME = r"exports\output.mp4"

# Video settings
VIDEO_FPS = 60.0
FONT_SIZE = 8
ENABLE_PREVIEW = True
PREVIEW_SCALE = 0.5  # 0.5 = 50% of original size

# Blob text parameters
BLOB_TEXT_FONT_SIZE = 24  # Size of blob numbers/words (increase for bigger text)
ENABLE_RANDOM_WORDS = True  # Set to False to use numbers instead
RANDOM_WORD_LIST = [
    "RIDI", "RIDINUUL", ".COM", "NUUL", "WWW"
]  # Words to randomly display on blobs

# Stylization parameters
EFFECT_SIZE_MULTIPLIER = 1.5  # Increase for larger blobs/effects
BLOB_OUTLINE_COLOR = (255, 0, 128)  # (R,G,B)
TEXT_COLOR = (255, 255, 255)  # (R,G,B)
LINE_COLOR = (255, 255, 255)  # (R,G,B)
INVERTED_EFFECT_OPACITY = 1.0  # 0.0 to 1.0

# Blob stabilization parameters
ENABLE_BLOB_STABILIZATION = True
BLOB_PERSISTENCE_FRAMES = 12  # Higher = smoother but more lag
BLOB_SMOOTHING_FACTOR = 0.6  # 0.0 = no smoothing, 1.0 = full smoothing

# Motion detection settings
MOTION_THRESHOLD = 25
MIN_BLOBS = 0.1
MAX_BLOBS_ALLOWED = 18

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def lerp(a, b, t):
    """Linear interpolation between a and b"""
    return a + (b - a) * t


def get_blob_text():
    """Generate random word or number for blob display"""
    if ENABLE_RANDOM_WORDS:
        return random.choice(RANDOM_WORD_LIST)
    else:
        return str(random.randint(10, 99))


def setup_audio_analysis(audio_path, video_fps):
    """Load and analyze audio, return RMS energy per frame"""
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = int(sr / video_fps)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms = rms / np.max(rms)
    return rms


def setup_blob_text_font(font_path, font_size):
    """Create font for blob text display"""
    return ImageFont.truetype(font_path, font_size)


def setup_blob_detector():
    """Create and configure blob detector"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 4000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    return cv2.SimpleBlobDetector_create(params)


def detect_motion(gray, prev_gray):
    """Detect motion between current and previous frame"""
    motion_points = []
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_points = np.column_stack(np.where(diff_thresh > 0))
    return motion_points


def calculate_blob_count(volume):
    """Calculate number of blobs based on audio volume"""
    num_blobs = int(MIN_BLOBS + (volume ** 1.5) * (MAX_BLOBS_ALLOWED - MIN_BLOBS))
    return min(num_blobs, MAX_BLOBS_ALLOWED)


def create_keypoints_from_motion(motion_points, num_blobs):
    """Create keypoints from motion detection"""
    keypoints = []
    if len(motion_points) > 0:
        for _ in range(num_blobs):
            yx = motion_points[random.randint(0, len(motion_points) - 1)]
            kp = cv2.KeyPoint(float(yx[1]), float(yx[0]), 20)
            keypoints.append(kp)
    return keypoints


def stabilize_blobs(keypoints, prev_blobs):
    """Smooth and persist blobs over time"""
    blobs = []
    used_prev = set()

    for i, kp in enumerate(keypoints):
        pt = np.array([kp.pt[0], kp.pt[1]])
        size = kp.size

        # Find closest previous blob
        min_dist = float('inf')
        min_idx = -1
        for j, prev in enumerate(prev_blobs):
            if j in used_prev:
                continue
            dist = np.linalg.norm(pt - prev['pt'])
            if dist < min_dist:
                min_dist = dist
                min_idx = j

        if min_idx != -1 and min_dist < 40:  # Match threshold
            prev = prev_blobs[min_idx]
            used_prev.add(min_idx)
            smoothed_pt = lerp(prev['pt'], pt, 1.0 - BLOB_SMOOTHING_FACTOR)
            smoothed_size = lerp(prev['size'], size, 1.0 - BLOB_SMOOTHING_FACTOR)
            blobs.append({
                'pt': smoothed_pt,
                'size': smoothed_size,
                'id': prev['id'],
                'persist': BLOB_PERSISTENCE_FRAMES,
                'value': prev['value']
            })
        else:
            # New blob with random text
            blobs.append({
                'pt': pt,
                'size': size,
                'id': i,
                'persist': BLOB_PERSISTENCE_FRAMES,
                'value': get_blob_text()
            })

    # Decrement persistence for unmatched previous blobs
    for j, prev in enumerate(prev_blobs):
        if j not in used_prev and prev['persist'] > 0:
            blobs.append({
                'pt': prev['pt'],
                'size': prev['size'],
                'id': prev['id'],
                'persist': prev['persist'] - 1,
                'value': prev['value']
            })

    # Remove blobs that have expired
    blobs = [b for b in blobs if b['persist'] > 0]
    return blobs


def draw_blobs(pil_frame, draw, blobs, blob_ids, frame_w, frame_h, volume, 
               inverted_pil, blob_text_font):
    """Draw blobs on the frame"""
    blob_polygons = {}

    for i, blob in enumerate(blobs):
        x, y = int(blob['pt'][0]), int(blob['pt'][1])
        base_size = int(blob['size'] // 2)
        size = int(base_size * (1 + 0.5 * volume) * EFFECT_SIZE_MULTIPLIER)
        w = max(10, size + random.randint(-6, 6))
        h = max(10, size + random.randint(-6, 6))
        points = [(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)]
        blob_polygons[i] = points

        # Inverted effect
        if random.random() < (0.15 + 0.45 * volume) * INVERTED_EFFECT_OPACITY:
            mask = Image.new("L", (frame_w, frame_h), 0)
            ImageDraw.Draw(mask).polygon(points, fill=255)
            pil_frame.paste(inverted_pil, mask=mask)

        # Draw outline and text
        draw.polygon(points, outline=BLOB_OUTLINE_COLOR)
        text_value = blob_ids[i]['value']
        # Center text on blob
        bbox = draw.textbbox((0, 0), text_value, font=blob_text_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x - text_w // 2
        text_y = y - text_h // 2
        draw.text((text_x, text_y), text_value, font=blob_text_font, fill=TEXT_COLOR)

    return blob_polygons


def draw_connections(draw, blobs, blob_polygons, anchor_blob_id):
    """Draw lines connecting blobs"""
    connections = {i: 0 for i in range(len(blobs))}

    if blobs and anchor_blob_id is not None:
        for i, blob in enumerate(blobs):
            if i == anchor_blob_id:
                continue

            available_hosts = [idx for idx, count in connections.items() if count < 2 and idx != i]
            if not available_hosts:
                continue

            host_idx = random.choice(available_hosts)
            connections[host_idx] += 1
            host_blob = blobs[host_idx]
            hx, hy = int(host_blob['pt'][0]), int(host_blob['pt'][1])

            polygon = blob_polygons.get(i)
            if not polygon:
                continue

            closest_point = min(polygon, key=lambda pt: (pt[0] - hx) ** 2 + (pt[1] - hy) ** 2)

            if random.random() < 0.2:
                cx = (hx + closest_point[0]) // 2 + random.randint(-100, 100)
                cy = (hy + closest_point[1]) // 2 + random.randint(-100, 100)
                curve_points = [(
                    int((1 - t) ** 2 * hx + 2 * (1 - t) * t * cx + t ** 2 * closest_point[0]),
                    int((1 - t) ** 2 * hy + 2 * (1 - t) * t * cy + t ** 2 * closest_point[1])
                ) for t in np.linspace(0, 1, 20)]
                draw.line(curve_points, fill=LINE_COLOR, width=1)
            else:
                draw.line((hx, hy, closest_point[0], closest_point[1]), fill=LINE_COLOR, width=1)


def display_preview(result):
    """Display preview window at reduced size"""
    preview_w = int(result.shape[1] * PREVIEW_SCALE)
    preview_h = int(result.shape[0] * PREVIEW_SCALE)
    preview_frame = cv2.resize(result, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("anAutoblobber Preview", preview_frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')


# ============================================================================
# MAIN PROCESSING
# ============================================================================


def main():
    """Main processing pipeline"""
    print("Loading audio...")
    rms = setup_audio_analysis(AUDIO_PATH, VIDEO_FPS)
    max_audio_frame = len(rms)

    print("Setting up video...")
    ui_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    blob_text_font = setup_blob_text_font(FONT_PATH, BLOB_TEXT_FONT_SIZE)
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_w, frame_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_NAME, fourcc, VIDEO_FPS, frame_size)

    detector = setup_blob_detector()

    # State variables
    blob_ids = {}
    anchor_blob_id = None
    anchor_switch_time = time.time() + random.uniform(3, 5)
    prev_gray = None
    prev_blobs = []
    frame_idx = 0

    print("Processing frames...")

    try:
        while True:
            audio_frame = frame_idx
            if audio_frame >= max_audio_frame:
                break

            volume = rms[audio_frame]

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion detection
            motion_points = detect_motion(gray, prev_gray)
            prev_gray = gray.copy()

            # Blob detection
            keypoints = detector.detect(gray)
            keypoints = list(keypoints)
            random.shuffle(keypoints)

            num_blobs = calculate_blob_count(volume)

            # Combine motion and static blobs
            motion_kps = create_keypoints_from_motion(motion_points, num_blobs)
            keypoints = motion_kps[:num_blobs - 2] + keypoints[:2]

            # Apply stabilization
            if ENABLE_BLOB_STABILIZATION:
                blobs = stabilize_blobs(keypoints, prev_blobs)
                prev_blobs = blobs.copy()
            else:
                blobs = [{
                    'pt': np.array([kp.pt[0], kp.pt[1]]),
                    'size': kp.size,
                    'id': i,
                    'persist': 1,
                    'value': get_blob_text()
                } for i, kp in enumerate(keypoints)]
                prev_blobs = blobs.copy()

            # Prepare frame
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)

            # Update blob IDs
            current_time = time.time()
            for i, blob in enumerate(blobs):
                if i not in blob_ids or current_time > blob_ids[i].get('expires', 0):
                    blob_ids[i] = {
                        'value': blob['value'],
                        'expires': current_time + random.uniform(1.5, 4.0)
                    }

            # Update anchor blob
            if blobs:
                if anchor_blob_id is None or current_time > anchor_switch_time or anchor_blob_id >= len(blobs):
                    anchor_blob_id = random.randint(0, len(blobs) - 1)
                    anchor_switch_time = current_time + random.uniform(3, 6)

            # Create inverted frame
            inverted_frame = cv2.bitwise_not(frame)
            inverted_pil = Image.fromarray(cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2RGB))

            # Draw blobs and connections
            blob_polygons = draw_blobs(pil_frame, draw, blobs, blob_ids, frame_w, frame_h, 
                                       volume, inverted_pil, blob_text_font)
            draw_connections(draw, blobs, blob_polygons, anchor_blob_id)

            # Convert and write
            result = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            out.write(result)

            # Display preview
            if ENABLE_PREVIEW:
                if display_preview(result):
                    print("Preview interrupted by user.")
                    break

            # Debug info
            print(f"Frame {frame_idx}: {num_blobs:.1f} blobs")
            frame_idx += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        if ENABLE_PREVIEW:
            cv2.destroyAllWindows()
        print("Export complete. Exiting.")


if __name__ == "__main__":
    main()

