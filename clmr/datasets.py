class CropAudioDataset(Dataset):
    def __init__(self, dataset, audio_length):
        self.dataset = dataset
        self.audio_length = audio_length
        self.cache = {}  # Simple cache for processed audio
        
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        x, y = self.dataset[idx]
        if x.shape[-1] > self.audio_length:
            start = torch.randint(0, x.shape[-1] - self.audio_length, (1,))
            x = x[..., start:start + self.audio_length]
        elif x.shape[-1] < self.audio_length:
            x = F.pad(x, (0, self.audio_length - x.shape[-1]))
            
        self.cache[idx] = (x, y)
        return x, y 