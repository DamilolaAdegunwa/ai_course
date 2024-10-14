import torchaudio
from speechbrain.pretrained import Tacotron2, HifiGan

# Initialize the pre-trained models
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir")
hifi_gan = HifiGan.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir")


# Function to synthesize speech
def synthesize_speech(text):
    mel_output, mel_length, _ = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save the speech to a file
    torchaudio.save('output.wav', waveforms.squeeze(1), 22050)

    print("Speech generated successfully and saved as 'output.wav'.")


# Test the TTS function
text = "The quick brown fox jumps over the lazy dog."
synthesize_speech(text)
