# Audio, Video, and Text Analysis

A Python project for analyzing audio, video, and text data.

## Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/audio_video_text_analysis.git
cd audio_video_text_analysis
```

2. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Project Structure

```
.
├── images/                                 # Contain images for face recognition     
├── detect_expression_video.py              # Script to detect expressions from faces
|__ recognize_expression_video.py           # Script to recognize faces and its expressions
├── venv/                                   # Virtual environment
├── requirements.txt                        # Project dependencies
└── README.md                               # This file
```

## Usage

In order to execute the scripts you need to provide the video and the images. 

### Images
A set of images of faces present in the video labeled with a name

### Video
A video called video.mp4 containing faces and expressions

## Running the scripts


```shell
python recognize_expression_video.py
```


## License

*Your license of choice* 