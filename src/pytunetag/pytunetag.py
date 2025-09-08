import os, torch, numpy as np
from pathlib import Path
import torch.nn.functional as F

from pytunetag.pytunetag.mtrpp.utils.eval_utils import load_ttmr_pp
from pytunetag.pytunetag.mtrpp.utils.audio_utils import load_audio, STR_CH_FIRST
from pydub import AudioSegment

from mutagen.id3 import ID3, TCON


class pyTuneTag:
    def __init__(self, genres:list[str], model_src:Path):

        finished_pth = Path(os.path.join(model_src, "finished.flag"))
        
        if not finished_pth.exists():
            torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.pth', model_src + 'best.pth')
            torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.yaml', model_src + 'hparams.yaml')
            
            finished_pth.touch()
            
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.model, sr, duration = load_ttmr_pp(model_src, model_types="best")
        self.model.to(self.device)

        self.sr = sr
        self.n_samples = int(self.sr * duration)
        
        self.genre_embs = {}
        for genre in genres:
            self.genre_embs[genre] = self.get_text_embedding(str(genre))

    def load_wav(self, audio_path):
        audio, _ = load_audio(
            path=audio_path,
            ch_format=STR_CH_FIRST,
            sample_rate=22050,
            downmix_to_mono=True
        )
        
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        
        if len(audio) < self.n_samples:
            return None

        ceil = audio.shape[-1] // self.n_samples
        audio = audio[:ceil * self.n_samples]
        return torch.from_numpy(np.stack(np.split(audio, ceil)).astype(np.float32))

    def load_mp3(self, audio_path:Path) -> torch.Tensor | None:
        audio:AudioSegment = AudioSegment.from_mp3(audio_path)        
        audio = audio.set_channels(1).set_frame_rate(self.sr)
        audio:np.ndarray = np.array(audio.get_array_of_samples(), dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
            
        if len(audio) < self.n_samples:
            return None

        ceil = audio.shape[-1] // self.n_samples
        audio = audio[:ceil * self.n_samples]
        return torch.from_numpy(np.stack(np.split(audio, ceil)).astype(np.float32))
        
    def get_audio_embedding(self, audio_path:Path) -> torch.Tensor | None:
        if str(audio_path).endswith(".mp3"):
            audio = self.load_mp3(audio_path)
        else:
            audio = self.load_wav(audio_path)

        if audio is None:
            return None # Doesn't exist / too short

        with torch.no_grad():
            z_audio:torch.Tensor = self.model.audio_forward(audio.to(self.device))
        z_audio = z_audio.mean(0).detach().cpu()
        return z_audio.float()

    def get_text_embedding(self, text:str) -> torch.Tensor:
        with torch.no_grad():
            z_tag:torch.Tensor = self.model.text_forward([text])    
        z_tag = z_tag.squeeze(0).detach().cpu()
        return z_tag.float()
    
    def set_meta(self, mp3_path:Path, genres:list[str]):
        audio = ID3(mp3_path)
        genre_string = '\x00'.join(genres)        
        audio['TCON'] = TCON(encoding=3, text=genre_string)
        audio.save()
        if 'TIT2' in audio and 'TPE1' in audio:
            return f"{audio['TPE1'].text[0]} - {audio['TIT2'].text[0]}"

        return None
    
    def __call__(self, target_mp3:Path):
        z_audio = self.get_audio_embedding(target_mp3)
        
        if z_audio is None:
            return None
        
        similarities = []
        for genre, z_tag in self.genre_embs.items():
            if z_audio.shape != z_tag.shape:
                continue
            
            similarity = F.cosine_similarity(z_audio.unsqueeze(0), z_tag.unsqueeze(0), dim=1).item()
            #similarity = np.linalg.norm(z_audio-z_tag)
            similarities.append((genre, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)