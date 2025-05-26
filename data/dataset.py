import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

# ==== Naive Method ====
class AudioDataset(Dataset):
    """
    Loading audio data from the time domain 
    """
    def __init__(self, root_dir):
        # decide type of audio files - .wav/.flac
        files = Path(root_dir).glob("*.wav")
        self.items = [(f, int(f.name.split("-")[-1].replace(".wav", ""))) for f in files]   # file, class
        self.length = len(self.items)

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        # load audio files - .wav or .flac
        audio_tensor, _ = torchaudio.load(filename)
        return audio_tensor, label

    def __len__(self):
        return self.length

# ==== Faster Method ==== 
"""
First, precompute the spectograms (freq-domain), save them, 
Then just load images since loading them are much more optimal
"""
def precompute_spectrograms(path, dpi=50):
    files = Path(path).glob('*.wav')

    for filename in tqdm(files, desc="Computing spectrograms"):
        output_path = filename.parent / f"spec_{dpi}_{filename.name}.png"
        if output_path.exists():
            continue
            
        audio_tensor, sample_rate = librosa.load(str(filename), sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio_tensor, sr=sample_rate)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=dpi)
        plt.close()

# Now, simply load images
class AudioDatasetSpectogram(Dataset):
    def __init__(self, root_dir, dpi=50, transforms=None):
        super().__init__()
        files = Path(root_dir).glob(f'spec_{dpi}*.wav.png')
        self.items = [(f, int(f.name.split('-')[-1].replace(".wav.png", ""))) for f in files]   # directly images
        self.length = len(self.items)

        if transforms is None:
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        image = Image.open(filename)
        return self.transforms(image), label

    def __len__(self):
        return self.length
    

if __name__ == "__main__":
    # dataset_time_domain = AudioDataset("../audio")
    # Firstly, one-time computation of spectrograms
    # precompute_spectrograms("../audio/train", dpi=50)
    # precompute_spectrograms("../audio/test", dpi=50)
    # precompute_spectrograms("../audio/valid", dpi=50)
    # Now, load spectrograms as images directly
    train_dataset_mel_specs = AudioDatasetSpectogram("../audio/train")
    valid_dataset_mel_specs = AudioDatasetSpectogram("../audio/valid")
    test_dataset_mel_specs = AudioDatasetSpectogram("../audio/test")

    # toy sample
    tensor, label = train_dataset_mel_specs[0]
    print(tensor.shape, label)  # [4, 200, 500] 14
