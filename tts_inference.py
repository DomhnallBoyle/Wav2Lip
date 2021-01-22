import argparse
import importlib
import os
import re
import subprocess
import time

import audio
import cv2
import numpy as np
import pyttsx3
import torch
from inference import device, load_model, datagen, mel_step_size
from tqdm import tqdm

MAX_FILE_LENGTH = 250
SRAVI_PHRASES = [
    "What's the plan?",
    "I feel depressed",
    "Call my family",
    "I'm hot",
    "I'm cold",
    "I feel anxious",
    "What time is it?",
    "I don't want that",
    "How am I doing?",
    "I need the bathroom",
    "I'm comfortable",
    "I'm thirsty",
    "It's too bright",
    "I'm in pain",
    "Move me",
    "It's too noisy",
    "Doctor",
    "I'm hungry",
    "Can I have a cough?",
    "I am scared"
]
ABS_FILE_PATH = os.path.abspath(__file__)
DIR_NAME = os.path.dirname(ABS_FILE_PATH)
TTS_SAVE_DIRECTORY = os.path.join(DIR_NAME, 'tts')
VIDEO_SAVE_DIRECTORY = os.path.join(DIR_NAME, 'output_videos')
for dir in [TTS_SAVE_DIRECTORY, VIDEO_SAVE_DIRECTORY]:
    if not os.path.exists(dir):
        os.makedirs(dir)


def phrase_to_output(phrase):
    s = str(phrase).lower().strip().replace(' ', '_')

    return re.sub(r'[\?\!]', '', s)


def shorten_filename(phrase):
    return '_'.join([word[0] for word in phrase.split(' ')])


def generate_tts(phrases, rate):
    for phrase in tqdm(phrases):
        wav_output = os.path.join(TTS_SAVE_DIRECTORY,
                                  f'{phrase_to_output(phrase)}.wav')
        if len(wav_output) > MAX_FILE_LENGTH:
            wav_output = os.path.join(TTS_SAVE_DIRECTORY,
                                      f'{shorten_filename(phrase)}.wav')
        if not os.path.exists(wav_output):
            # print('Running TTS for:', phrase)
            while not os.path.exists(wav_output):
                try:
                    importlib.reload(pyttsx3)
                    engine = pyttsx3.init()
                    engine.setProperty('rate', rate)
                    engine.save_to_file(phrase, wav_output.replace("'", "\\'"))
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(phrase, 'TTS failed', e)
                    os.remove(wav_output)
                time.sleep(1)


def get_video_rotation(video_path):
    cmd = f'ffmpeg -i {video_path}'

    p = subprocess.Popen(
        cmd.split(' '),
        stderr=subprocess.PIPE,
        close_fds=True
    )
    stdout, stderr = p.communicate()

    try:
        reo_rotation = re.compile('rotate\s+:\s(\d+)')
        match_rotation = reo_rotation.search(str(stderr))
        rotation = match_rotation.groups()[0]
    except AttributeError:
        print(f'Rotation not found: {video_path}')
        return 0

    return int(rotation)


def fix_frame_rotation(image, rotation):
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def generate_new_video(args, input_video, input_audio, output_file):
    video_stream = cv2.VideoCapture(input_video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    rotation = get_video_rotation(input_video)

    print('Reading video frames...')

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor,
                                       frame.shape[0] // args.resize_factor))

        frame = fix_frame_rotation(frame, rotation)

        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if not input_audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = "ffmpeg -y -i {} -strict -2 {}".format(input_audio,
                                                         'temp/temp.wav')

        subprocess.call(command, shell=True)
        input_audio = 'temp/temp.wav'

    wav = audio.load_wav(input_audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(args, full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in\
            enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                  (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    input_audio = input_audio.replace("'", "\\'")
    output_file = output_file.replace("'", "\\'")
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'\
        .format(input_audio, 'temp/result.avi', output_file)
    print(command)
    subprocess.call(command, shell=True)


def main(args):
    args.img_size = 96
    phrases = args.input_phrases if args.input_phrases else SRAVI_PHRASES

    if args.tts:
        generate_tts(phrases, args.rate)

    num_sessions = len(args.input_videos)

    for session_number in range(num_sessions):
        input_video_path = args.input_videos[session_number]
        for phrase in phrases:
            audio_input = \
                os.path.join(TTS_SAVE_DIRECTORY,
                             f'{phrase_to_output(phrase)}.wav')
            if len(audio_input) > MAX_FILE_LENGTH:
                audio_input = os.path.join(TTS_SAVE_DIRECTORY,
                                           f'{shorten_filename(phrase)}.wav')

            output_video_file = \
                os.path.join(VIDEO_SAVE_DIRECTORY,
                             f'{phrase_to_output(phrase)}_{session_number+1}.mp4')
            if len(output_video_file) > MAX_FILE_LENGTH:
                output_video_file = \
                    os.path.join(VIDEO_SAVE_DIRECTORY,
                                 f'{shorten_filename(phrase)}_{session_number+1}.mp4')

            if os.path.exists(output_video_file):
                continue
            print('Generating new video:', output_video_file, audio_input)
            generate_new_video(args, input_video_path, audio_input,
                               output_video_file)
            if not os.path.exists(output_video_file):
                print(phrase, 'failed')
                exit(1)

            with open(os.path.join(VIDEO_SAVE_DIRECTORY, 'groundtruth.csv'),
                      'a') as f:
                f.write(f'{os.path.basename(output_video_file)},{phrase},{phrase}\n')


def str_list(s):
    return s.split(',')


def file_list(s):
    with open(s, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_videos', type=str_list)
    parser.add_argument('--input_phrases', type=file_list, default=None)
    parser.add_argument('--tts', action='store_true')
    parser.add_argument('--rate', type=int, default=180)  # lower = slower

    parser.add_argument('--checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from',
                        required=True)

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference',
                        default=False)
    parser.add_argument('--fps', type=float,
                        help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int,
                        help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    main(parser.parse_args())
