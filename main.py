from dotenv import load_dotenv
from stt import WhisperSTT


load_dotenv()
# Having such a lightweight implementation of the model allows to easily integrate it in different platforms
# and applications. As an example, here is a video of running the model on an iPhone 13 device - fully offline, on-device
wSTT = WhisperSTT("turn")
wSTT.listen()

# vSTT = VoskSTT()
# vSTT.listen()


