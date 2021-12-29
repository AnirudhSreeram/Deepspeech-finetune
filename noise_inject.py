import argparse
from dataclasses import dataclass

import torch
from scipy.io.wavfile import write

from deepspeech_pytorch.loader.data_loader import load_audio, NoiseInjection

@dataclass
class aug:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = '/data/asreeram/deepspeech.pytorch/noise_dir/WGN' #'/data/asreeram/deepspeech.pytorch/noise_dir' # '/home/asreeram/Workspace/deepspeech.pytorch/noise'  # Directory to inject noise into audio. If default, noise Inject not added
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0
    #noise_levels: noise_levels = (noise_min,noise_max)
    perceptual_noise: bool = False
    fft_len: int = 512
    win_length: int = 256 
    hop_length: int = 128
    sample_rate: int =16000

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', default=16000, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0,
                    help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

noise_injector = NoiseInjection()
data = load_audio(args.input_path)
mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level,aug)
mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
write(filename=args.output_path,
      data=mixed_data.numpy(),
      rate=args.sample_rate)
print('Saved mixed file to %s' % args.output_path)
