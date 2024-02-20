import numpy as np
import torch
# import librosa
# from pydub import AudioSegment, generators

# Add white noise to audio
def add_white_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

# Add clipping to audio
def add_clipping(audio, clipping_level=0.005):
    y = audio[0,:]

    # Generate a noise sample consisting of values that are a little higer or lower than a few randomly selected values in the original data. 
    noise_sample = np.random.default_rng().uniform(0.2*min(y), 0.3*max(y), int(clipping_level*len(y)))

    # Generate an array of zeros with a size that is the difference of the sizes of the original data an the noise sample.
    zeros = np.zeros(len(y) - len(noise_sample))

    # Add the noise sample to the zeros array to obtain the final noise with the same shape as that of the original data.
    noise = np.concatenate([noise_sample, zeros])

    # Shuffle the values in the noise to make sure the values are randomly placed.
    np.random.shuffle(noise)

    # Obtain data with the noise added.
    return audio + torch.tensor(noise).float()

# Add quantization noise to audio
def add_quantization_noise(audio, noise_percentage=0.2, quantization_level=0.005):
    y = audio[0,:]

    # Get the size of the independent variable
    y_size = len(y)

    # Determine the size of the noise based on the noise precentage
    noise_size = int(noise_percentage * y_size)

    # Randomly select indices for adding noise.
    random_indices = np.random.choice(y_size, noise_size)

    # Create a copy of the original data that serves as a template for the noised data.
    y_noised = y.numpy().copy()

    # Round-off values of the templated noised data at random indices, to obtain the final noised data.
    y_noised[random_indices] = np.rint(y_noised[random_indices])

    return torch.tensor(y_noised).reshape((1, -1)).float()

# Add salt and pepper noise to audio
def add_salt_and_pepper_noise(audio, noise_percentage=0.001, noise_level=0.005):
    # Get the image size (number of pixels in the image).
    y = audio[0,:]
    audio_size =  len(y)

    # Determine the size of the noise based on the noise precentage
    noise_size = int(noise_percentage*audio_size)

    # Randomly select indices for adding noise.
    random_indices = np.random.choice(audio_size, noise_size)

    # Create a noise list with random placements of min and max values of the image pixels.
    noise = np.random.choice([y.min(), y.max()], noise_size)
    noise = noise * noise_level

    y_noised = y.numpy().copy()

    # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
    y_noised.flat[random_indices] = noise
    
    return torch.tensor(y_noised).reshape((1, -1)).float()

# distort audio
def distort_audio(audio):
    # Randomly select a function to apply
    functions = {add_white_noise, add_clipping, add_quantization_noise, add_salt_and_pepper_noise}
    functions = np.random.choice(list(functions), np.random.randint(1, len(functions)), replace=False)

    # Iterate over the selected functions and apply them to the audio
    for func in functions:
        # Generate random parameters within certain ranges for each function
        if func == add_white_noise:
            noise_level = np.random.uniform(0.05, 0.4)
            audio = func(audio, noise_level)
        elif func == add_clipping:
            clipping_level = np.random.uniform(0.05, 0.4)
            audio = func(audio, clipping_level)
        elif func == add_quantization_noise:
            noise_percentage = np.random.uniform(0.2, 0.4)
            quantization_level = np.random.uniform(0.05, 0.4)
            audio = func(audio, noise_percentage, quantization_level)
        elif func == add_salt_and_pepper_noise:
            noise_percentage = np.random.uniform(0.05, 0.4)
            noise_level = np.random.uniform(0.05, 0.4)
            audio = func(audio, noise_percentage, noise_level)

    # Return the distorted audio
    return audio

# Test the distort_audio function
if __name__ == "__main__":
    audio = torch.randn(1, 8000)
    print(audio.shape)
    distorted_audio = distort_audio(audio)
    print(distorted_audio.shape)


# # Add pink noise to audio
# def add_pink_noise(audio, noise_level=0.005):


# # Add brown noise to audio
# def add_brown_noise(audio, noise_level=0.005):


# def change_speed(audio, speed_factor=1.0):
#     return librosa.effects.time_stretch(audio, speed_factor)

# def change_pitch(audio, pitch_factor=1.0):
#     return librosa.effects.pitch_shift(audio, sr=22050, n_steps=pitch_factor)

# def change_volume(audio, volume_factor=1.0):
#     return audio * volume_factor

# def add_hum(audio, hum_level=0.005):


# def add_echo(audio, echo_level=0.005):


# def add_reverb(audio, reverb_level=0.005):

# def add_high_pass_filter(audio, cutoff=0.5):
#     return librosa.effects.preemphasis(audio, coef=cutoff)

# def add_low_pass_filter(audio, cutoff=0.5):
#     return librosa.effects.preemphasis(audio, coef=cutoff)

# def add_band_pass_filter(audio, cutoff=0.5):
#     return librosa.effects.preemphasis(audio, coef=cutoff)

# def add_clipping(audio, clipping_level=0.005):

# def add_sample_rate_reduction(audio, reduction_factor=2):
#     return librosa.resample(audio, 22050, 22050//reduction_factor)

# def add_aliasing(audio, aliasing_level=0.005):

# def add_codec(audio, codec_level=0.005):

# def add_bit_reduction(audio, reduction_level=0.005):

# def add_bit_crusher(audio, crusher_level=0.005):

