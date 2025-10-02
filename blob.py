import cv2
import numpy as np
import random
import time
from PIL import ImageFont, ImageDraw, Image
import librosa
import pygame

# === CONFIG ===
font_path = r"static\ibmreg.ttf" # for custom fonts
video_path = r"static\video.mp4" #video to be blobbed
audio_path = r"static\song.wav" #audio to be blobbed to
output_name = r"exports\output.mp4" #export render
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
ENABLE_PREVIEW = True  # you can turn this off if you just want the exported render
font = ImageFont.truetype(font_path, 8) #change font here, you load the font in the root

# AUDIO ANALYSIS WITH LIBROSA
y, sr = librosa.load(audio_path, sr=None)
hop_length = int(sr / video_fps)
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
rms = rms / np.max(rms)

# START AUDIO
pygame.mixer.init()
pygame.mixer.music.load(audio_path)
pygame.mixer.music.play()
start_time = time.time()

# VIDEO SETUP
cap = cv2.VideoCapture(video_path)

# BLOB DETECTOR SETUP
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
# edit these value to indicate how many blobs get drawn at low vs. high volume
detector = cv2.SimpleBlobDetector_create(params)
blob_ids = {}
anchor_blob_id = None
anchor_switch_time = time.time() + random.uniform(3, 5)

# MOTION DETECTION SETUP
prev_gray = None
motion_thresh = 25  # sensitivity

# OUTPUT VIDEO WRITER
output_path = output_name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # you can use 'avc1' or 'XVID' if mp4v doesn't work
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, fourcc, video_fps, frame_size)

# BLOB PROCESSOR
while True:
    elapsed = time.time() - start_time
    audio_frame = int(elapsed * video_fps)

    # Audio Restarter
    if audio_frame >= len(rms):
        pygame.mixer.music.stop()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        start_time = time.time()
        pygame.mixer.music.play()
        continue
    volume = rms[audio_frame]

    # Video Restarter
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Motion detector
    motion_points = []
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_thresh = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
        motion_points = np.column_stack(np.where(diff_thresh > 0))
    prev_gray = gray.copy()

    # Blob detection
    keypoints = detector.detect(gray)
    keypoints = list(keypoints)
    random.shuffle(keypoints)

    # number of blobs is based on audio volume
    min_blobs = 0.1
    max_blobs_allowed = 18
    num_blobs = int(min_blobs + (volume**1.5) * (max_blobs_allowed - min_blobs))
    num_blobs = min(num_blobs, max_blobs_allowed)

    # Create motion blobs (majority)
    motion_kps = []
    if len(motion_points) > 0:
        for _ in range(num_blobs):
            yx = motion_points[random.randint(0, len(motion_points)-1)]
            kp = cv2.KeyPoint(float(yx[1]), float(yx[0]), 20)
            motion_kps.append(kp)
    # few static blobs for variety
    keypoints = motion_kps[:num_blobs-2] + keypoints[:2]

    # Convert to PIL frame
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)

    # Blob numbering
    current_time = time.time()
    for i, kp in enumerate(keypoints):
        if i not in blob_ids or current_time > blob_ids[i]['expires']:
            blob_ids[i] = {
                'value': random.randint(10, 99),
                'expires': current_time + random.uniform(1.5, 4.0)
            }

    # Anchor blob determiner
    if keypoints:
        if anchor_blob_id is None or current_time > anchor_switch_time or anchor_blob_id >= len(keypoints):
            anchor_blob_id = random.randint(0, len(keypoints)-1)
            anchor_switch_time = current_time + random.uniform(3, 6)

    # inverted frame setup
    blob_polygons = {}
    inverted_frame = cv2.bitwise_not(frame)
    inverted_pil = Image.fromarray(cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2RGB))

    # Blob drawer
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        base_size = int(kp.size // 2)
        size = int(base_size * (1 + 0.5 * volume))
        w, h = max(10, size + random.randint(-6,6)), max(10, size + random.randint(-6,6))
        points = [(x-w, y-h), (x+w, y-h), (x+w, y+h), (x-w, y+h)]
        blob_polygons[i] = points

        if random.random() < 0.15 + 0.45*volume:
            mask = Image.new("L", (frame_w, frame_h), 0)
            ImageDraw.Draw(mask).polygon(points, fill=255)
            pil_frame.paste(inverted_pil, mask=mask)

        r = int(150 + 105*volume)
        draw.polygon(points, outline=(r,0,0))
        draw.text((x+5,y-10), str(blob_ids[i]['value']), font=font, fill=(255,255,255))

    # Line drawing: limit 2 connections per blob host
    connections = {i:0 for i in range(len(keypoints))}  # track how many connections each blob has
    if keypoints:
        anchor_kp = keypoints[anchor_blob_id]
        ax, ay = int(anchor_kp.pt[0]), int(anchor_kp.pt[1])

        for i, kp in enumerate(keypoints):
            if i == anchor_blob_id:
                continue
            # Find a host blob that has <2 connections
            available_hosts = [idx for idx, count in connections.items() if count < 2 and idx != i]
            if not available_hosts:
                continue
            host_idx = random.choice(available_hosts)
            connections[host_idx] += 1
            host_kp = keypoints[host_idx]
            hx, hy = int(host_kp.pt[0]), int(host_kp.pt[1])

            polygon = blob_polygons.get(i)
            if not polygon:
                continue
            closest_point = min(polygon, key=lambda pt: (pt[0]-hx)**2 + (pt[1]-hy)**2)

            if random.random() < 0.2:
                cx = (hx + closest_point[0])//2 + random.randint(-100,100)
                cy = (hy + closest_point[1])//2 + random.randint(-100,100)
                curve_points = [(int((1-t)**2*hx + 2*(1-t)*t*cx + t**2*closest_point[0]),
                                 int((1-t)**2*hy + 2*(1-t)*t*cy + t**2*closest_point[1]))
                                for t in np.linspace(0,1,20)]
                draw.line(curve_points, fill=(255,255,255), width=1)
            else:
                draw.line((hx, hy, closest_point[0], closest_point[1]), fill=(255,255,255), width=1)

    #you'll see this in the console, how many blobs are being drawn per frame so you can see how the blob
    #drawer is reacting to the audio you chose
    print(f"Blobs on screen: {num_blobs:.3f}")


    # display frame
    result = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    out.write(result)  # Save current frame to output file

    if ENABLE_PREVIEW:
        cv2.imshow("anAutoblobber", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cleanup
cap.release()
out.release()  # Release video writer
pygame.mixer.quit()
cv2.destroyAllWindows()

