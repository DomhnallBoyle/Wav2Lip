import os

from inference import generate_new_video, TTS_SAVE_DIRECTORY, \
    VIDEO_SAVE_DIRECTORY

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from gtts import gTTS

VIDEO_INPUT = os.path.join(VIDEO_SAVE_DIRECTORY, 'input.mp4')
WAV_OUTPUT = os.path.join(TTS_SAVE_DIRECTORY, 'tts.wav')
VIDEO_OUTPUT = os.path.join(VIDEO_SAVE_DIRECTORY, 'output.mp4')

app = FastAPI()


# default arguments
class Namespace:
    def __init__(self):
        self.checkpoint_path = 'checkpoints/wav2lip.pth'
        self.static = False
        self.fps = 25
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.nosmooth = 'False'
        self.img_size = 96


args = Namespace()


@app.post('/generate')
async def generate(phrase: str,
                   video_file: UploadFile = File(...),
                   slow: bool = False,
                   accent: str = 'co.uk'):

    with open(VIDEO_INPUT, "wb+") as f:
        f.write(video_file.file.read())

    tts = gTTS(phrase, lang='en', tld=accent, slow=slow)
    tts.save(WAV_OUTPUT)

    generate_new_video(
        args,
        input_video=VIDEO_INPUT,
        input_audio=WAV_OUTPUT,
        output_file=VIDEO_OUTPUT
    )

    return FileResponse(VIDEO_OUTPUT)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
