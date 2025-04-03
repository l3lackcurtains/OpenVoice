import os
import torch
import time
import functools
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

class VoiceGenerator:
    def __init__(self):
        # Basic setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = 'outputs_v2'
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models
        self.tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=self.device)
        self.tone_color_converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')
        self.model = TTS(language='EN', device=self.device)

        # Compile models if PyTorch >= 2.0
        if hasattr(torch, 'compile'):
            self.tone_color_converter.model = torch.compile(self.tone_color_converter.model)
            self.model.model = torch.compile(self.model.model)

        # Cache reference speaker embedding
        self.reference_speaker = 'resources/demo_speaker2.mp3'
        self.target_se = self._cache_reference_embedding()
        
        # Cache source speaker embedding
        self.speaker_key = 'en-us'
        self.source_se = self._cache_source_embedding()

    @functools.lru_cache(maxsize=None)
    def _cache_reference_embedding(self):
        target_se, _ = se_extractor.get_se(self.reference_speaker, self.tone_color_converter, vad=True)
        return target_se

    @functools.lru_cache(maxsize=None)
    def _cache_source_embedding(self):
        return torch.load(
            f'checkpoints_v2/base_speakers/ses/{self.speaker_key}.pth', 
            map_location=self.device
        )

    def generate_speech(self, text: str, speed: float = 1.0):
        start_time = time.time()

        # Generate speech
        src_path = f'{self.output_dir}/tmp.wav'
        self.model.tts_to_file(text, 0, src_path, speed=speed)

        # Convert tone color
        save_path = f'{self.output_dir}/output_v2_{self.speaker_key}.wav'
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path=save_path,
            message="@MyShell"
        )

        print(f"Speech generation time: {time.time() - start_time:.2f} seconds")
        return save_path

def main():
    # Create persistent generator instance
    generator = VoiceGenerator()
    
    # Example usage
    text = "Hey man, i know you better than that girl.. all the people in the world."
    output_path = generator.generate_speech(text)
    print(f"Generated speech saved to: {output_path}")

if __name__ == "__main__":
    main()
