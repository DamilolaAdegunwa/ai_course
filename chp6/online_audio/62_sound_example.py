import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import reverb

# Sound source location (example)
sound_location = {"x": 0, "y": 1, "z": 0}


# Calculate distance between listener and sound source
def calculate_distance_to_sound(position, sound_location):
    return np.sqrt((position['x'] - sound_location['x']) ** 2 +
                   (position['y'] - sound_location['y']) ** 2 +
                   (position['z'] - sound_location['z']) ** 2)


# Modify sound volume based on distance
def modify_sound_volume(sound, volume_factor):
    return sound + (volume_factor * 10 - 10)  # Adjusts volume in dB


# Adjust spatial audio based on position and sound source
def adjust_spatial_audio(position, sound):
    distance = calculate_distance_to_sound(position, sound_location)
    volume = 1 / (distance + 1)  # Simple inverse distance attenuation
    return modify_sound_volume(sound, volume)


# Apply reverb to generated sound
def apply_reverb(sound, reverb_properties):
    return reverb(sound)


# Match generated sound to environment with reverb
def match_generated_sound_to_environment(generated_sound, reverb_properties):
    modified_sound = apply_reverb(generated_sound, reverb_properties)
    return modified_sound


# Example usage
if __name__ == "__main__":
    # Load an audio file for testing
    sound = AudioSegment.from_file("example.wav")

    # Adjust for spatial audio
    position = {"x": 2, "y": 0, "z": 0}  # Listener's position
    spatial_sound = adjust_spatial_audio(position, sound)

    # Apply reverb
    reverb_properties = {"intensity": 0.7}  # Example property
    modified_sound = match_generated_sound_to_environment(spatial_sound, reverb_properties)

    # Play the modified sound
    play(modified_sound)
