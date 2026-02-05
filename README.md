# EasyBlobTracker

A Simple Audio Reactive Blob Visualization Tool that generates dynamic blobs synchronized with video and audio.

## Tech Stack

Python 3.12 | OpenCV | Librosa | Pillow | NumPy

## Features

- Audio Reactive blobs that scale with volume
- Motion detection integration
- Smooth blob stabilization and interpolation
- Customizable colors, sizes, and text
- Random words or numbers on blobs
- Connecting lines between blobs
- 60fps output optimization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python blob.py
```

Required files:
- `static/video.mp4` - Input video
- `static/song.wav` - Audio file
- `static/ibmreg.ttf` - Font file

Output: `exports/output.mp4`

## Example Videos

Add sample videos to your repo in `samples/` folder:

### Input Video
<video width="640" height="480" controls>
  <source src="exports/output.mp4" type="video/mp4">
</video>

### Output Video (With Blobs)
<video width="640" height="480" controls>
  <source src="static/video.mp4" type="video/mp4">
</video>

## Configuration

Key parameters in `blob.py`:

```python
VIDEO_FPS = 60.0
BLOB_TEXT_FONT_SIZE = 24
ENABLE_RANDOM_WORDS = True
EFFECT_SIZE_MULTIPLIER = 1.5
BLOB_OUTLINE_COLOR = (255, 0, 128)
BLOB_PERSISTENCE_FRAMES = 12
BLOB_SMOOTHING_FACTOR = 0.6
ENABLE_PREVIEW = True
```

## Customization Examples

Bigger blobs:
```python
BLOB_TEXT_FONT_SIZE = 48
EFFECT_SIZE_MULTIPLIER = 2.5
```

Smooth motion:
```python
BLOB_PERSISTENCE_FRAMES = 20
BLOB_SMOOTHING_FACTOR = 0.8
```

Fast response:
```python
BLOB_PERSISTENCE_FRAMES = 6
BLOB_SMOOTHING_FACTOR = 0.3
```


## License

MIT
