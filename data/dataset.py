import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

class AudioDataset(Dataset):
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
    
if __name__ == "__main__":
    dataset = AudioDataset("../audio")
    tensor, label = list(dataset)[0]
    # toy sample
    print(tensor, label)
