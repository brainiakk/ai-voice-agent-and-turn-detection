import json
import os
import subprocess
import time
import pyaudio
import speech_recognition as sr
import numpy as np
from scipy.spatial.distance import cdist
import torchaudio
import vosk
from pyannote.audio import Pipeline, Inference, Model
import torch

from smartturn.inference import predict_endpoint
from turnsense.infer import infer_from_file

# class VoskSTT:
#     def __init__(self):
#         self.speakerID = [-0.684831, 0.920557, 0.410813, 0.496618, -0.470445, 0.133223, 0.571884, 0.273272, 0.800151, 1.053392, 0.362485, -0.928948, -0.708939, -0.745662, 1.100233, 1.200025, -0.845873, 1.499482, 0.376734, -1.500299, -1.116465, 0.374384, 0.719967, -0.134071, -0.826093, -0.030378, 0.305073, 0.237386, 1.634479, -0.531842, -0.845318, -0.138725, -0.957128, 0.208981, 0.515501, -0.425019, 1.079459, 0.099933, 0.049681, -0.744659, -0.820231, 0.365551, 0.932675, 0.112584, -1.670888, 0.936135, 0.516301, -0.141263, 2.357004, 1.036008, -0.027368, -0.827204, 0.839867, 0.589744, 2.426642, -0.603773, 0.719184, -0.86386, -0.681348, 1.041839, 0.578906, -0.384341, -0.704846, 0.622501, -0.10019, 0.390479, -0.605709, 1.595512, 1.312619, 1.119397, 1.243152, -0.695057, -0.124289, 2.406557, 1.326846, 2.556342, -0.899463, -0.901283, 0.157395, 0.013824, -1.443996, 0.061373, 2.340236, -0.568972, -1.349781, 0.824917, -1.548339, -0.176583, -0.559307, 0.152892, 2.00335, -0.292749, -1.094699, 1.119134, 1.1468, 0.123479, 0.901928, -0.294221, -1.106891, -0.613252, -0.383081, -1.101079, 0.104254, -1.949001, 1.095181, -0.171511, 0.93907, 1.648136, -1.468865, 1.240855, 0.364059, -1.754388, 1.660288, -0.671913, -1.210148, 0.392263, 0.132216, -0.332478, -1.971254, 1.15104, -1.189939, -0.955338, -0.800723, -0.692481, -0.962559, -0.185366, -0.909458, 0.344498]

#     def listen(self):
#         print('Listening (mode: VoskSTT)...')

#         model = vosk.Model("models/vosk/vosk-model-en-us-0.22")

#         speaker_model = vosk.SpkModel("models/vosk/vosk-model-spk-0.4")

#         # Create recognizer with both speech and speaker model
#         recognizer = vosk.KaldiRecognizer(model, 16000)
#         recognizer.SetSpkModel(speaker_model)
#         stream = pyaudio.PyAudio().open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=4000)

#         while True:
#             text = ""
#             data = stream.read(4096, exception_on_overflow=False)


#             if recognizer.AcceptWaveform(data):
#                 result = json.loads(recognizer.Result())
#                 speaker_vector = result.get("spk", None)
#                 # print(speaker_vector)
#                 text = result.get("text", None)
#                 if speaker_vector and self.speaker_check(speaker_vector):
#                     if text:
#                         # Put recognized text and speaker vector into the queue
#                         print("(Vosk) You said:", text)


#     def cosine_dist(self, x, y):
#         nx = np.array(x)
#         ny = np.array(y)
#         return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

#     def speaker_check(self, speaker_vector):
#         avg_distance = self.cosine_dist(self.speakerID, speaker_vector)
#         print( "VoskSTT Cos Distance:", avg_distance)
#         if avg_distance < 0.7:
#             return True
#         return False


class WhisperSTT:
    def __init__(self, turn_type="smartturn"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddingmodel = Model.from_pretrained("pyannote/embedding", cache_dir="./models/pyanotte-model",
                                               token=os.getenv("HF_API_KEY"))
        self.inference = Inference(embeddingmodel, window="whole", device=self.device)
        self.main_speaker_embedding = self.inference("austin.wav")
        print(self.main_speaker_embedding)
        self.incomplete_turn_count = 0
        self.audio_past = None      # in-memory waveform tensor (1, T) of accumulated incomplete turns
        self.audio_past_sr = None

        # primary_model is the one that actually decides complete/incomplete and
        # drives process_audio. When use_only_primary is False, the other model
        # also runs on every chunk purely so both predictions can be compared
        # (e.g. on screen) - it never affects the turn-taking decision.
        self.primary_model = turn_type
        self.use_only_primary = True
        self.on_turn_status = None  # optional callback(dict) -> {"smartturn": {...}, "turnsense": {...}}

    def callback(self, recognizer, audio):
        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            # write audio to a RAW file
            
            transcribed_file = "transcribed-audio.wav"
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())
        
            start = time.time()
            output = self.turn_detection(transcribed_file)
            end = time.time()
            print('⏰ Listen Offline Transcription in %s seconds' % (end - start))

            clean_output = str(output).replace(" ", "").replace(".", "").replace(",", "").lower()
            if clean_output != "you" and clean_output != "":
                print("(Whisper) You Said: " + output)

        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    def listen(self):
        print('Listening (mode: WhisperCPP)...')
        try:
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                print("Say something!")

            stop_listening = r.listen_in_background(source, self.callback)
            while True:
                time.sleep(0.05)

        except Exception as e:
            print(f"Error while listening: {e}")
            stop_listening(wait_for_stop=False)
            return False

    
    def turn_detection(self, audio_path="transcribed-audio.wav"):
        # Check if the speaker is verified
        if not self.speaker_verified(audio_path):
            print("Speaker not verified")
            return ""

        # Load the current segment once. Past (still-incomplete) audio is kept as an
        # in-memory tensor rather than a file, so a long run of "incomplete" turns
        # doesn't re-read/re-write an ever-growing WAV file on every single cycle —
        # only accumulation (cheap torch.cat) happens per turn; disk is touched once,
        # when we actually need a file (TurnSense inference, or a finished transcription).
        current_waveform, current_sr = torchaudio.load(audio_path)
        if current_waveform.shape[0] > 1:
            current_waveform = current_waveform.mean(dim=0, keepdim=True)  # downmix to mono

        has_past = self.audio_past is not None
        if has_past:
            if self.audio_past_sr != current_sr:
                raise ValueError("Sample rates of past and present audio do not match.")
            combined_waveform = torch.cat((self.audio_past, current_waveform), dim=1)
        else:
            combined_waveform = current_waveform

        # Run whichever model(s) are needed: always the primary (it drives the
        # decision below); the secondary only when comparing both on screen.
        need_smartturn = self.primary_model == "smartturn" or not self.use_only_primary
        need_turnsense = self.primary_model == "turnsense" or not self.use_only_primary

        predictions = {}
        combined_path = None

        if need_smartturn:
            waveform = combined_waveform
            if current_sr != 16000:
                waveform = torchaudio.functional.resample(waveform, current_sr, 16000)
            audio_f32 = waveform.squeeze(0).numpy()
            predictions["smartturn"] = predict_endpoint(audio_f32)

        if need_turnsense:
            # TurnSense inference needs a file on disk.
            combined_path = self._save_combined(combined_waveform, current_sr)
            predictions["turnsense"] = infer_from_file(combined_path)

        if self.on_turn_status:
            self.on_turn_status(dict(predictions))

        predict = predictions[self.primary_model]

        if predict["prediction"] == "complete" and predict["probability"] > 0.6:
            print("Turn Complete")
            # self.incomplete_turn_count = 0
            final_path = combined_path or self._save_combined(combined_waveform, current_sr)
            self.audio_past = None
            self.audio_past_sr = None
            return self.process_audio(final_path)

        print("Turn Incomplete")
        self.audio_past = combined_waveform
        self.audio_past_sr = current_sr
        # self.incomplete_turn_count += 1
        return ""

    def _save_combined(self, waveform, sample_rate, path="transcribed-audio-past.wav"):
        torchaudio.save(path, waveform, sample_rate)
        return path

    def speaker_verified(self, speaker_wav) -> bool:
        speaker_embedding = self.inference(speaker_wav)
        distance = cdist(np.reshape(self.main_speaker_embedding, (1, -1)), np.reshape(speaker_embedding, (1, -1)), metric="cosine")[0, 0]

        # closer the distance value is to 0 the more likely the speaker is the same person.
        print( "WhisperSTT Cosine Distance:", distance)
        if distance < 0.7:
            return True

        return False

    def process_audio(self, wav_file, model_name="base.en"):
        """
        Processes an audio file using a specified model and returns the processed string.

        :param wav_file: Path to the WAV file
        :param model_name: Name of the model to use
        :return: Processed string output from the audio processing
        :raises: Exception if an error occurs during processing
        """
        

        model = f"./models/whisper.cpp/models/ggml-{model_name}.bin"

        # Check if the file exists
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file not found: {model} \n\nDownload a model with this command:\n\n> bash ./models/download-ggml-model.sh {model_name}\n\n")

        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        full_command = f"./models/whisper.cpp/build/bin/whisper-cli -m {model} -f {wav_file} -np -nt -fa -l en"

        # Execute the command
        process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Get the output and error (if any)
        output, error = process.communicate()

        if process.returncode != 0:
            raise Exception(f"Error processing audio: {error.decode('utf-8')}")

        # Process and return the output string
        decoded_str = output.decode('utf-8').strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', '').replace('[BLANK_AUDIO] ,', '').replace('[INAUDIBLE]', '').replace('[ Silence ]', '').replace('[ Silence ] ,', '').replace('[INAUDIBLE] ,', '').strip()

        return processed_str
