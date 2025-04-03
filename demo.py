import os
import torch
import time
import warnings
from transformers.utils import logging
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# Suppress transformer warnings for cleaner output
logging.set_verbosity_error()

class VoiceGenerator:
    """
    A class for generating voice outputs using OpenVoice and MeloTTS.
    Handles model initialization, caching, and speech generation.
    """
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = 'outputs_v2'
        self.speaker_key = 'en-us'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache paths for faster access
        self.temp_path = os.path.join(self.output_dir, 'tmp.wav')
        self.output_path = os.path.join(self.output_dir, f'output_v2_{self.speaker_key}.wav')
        
        # Initialize models with optimizations
        self._initialize_models()
        
        # Pre-load and cache source embedding
        self.source_se = torch.load(
            f'checkpoints_v2/base_speakers/ses/{self.speaker_key}.pth',
            map_location=self.device
        )
        
        # Cache target SE extractor settings
        self.se_extract_params = {'vad': True}

    @torch.inference_mode()  # Faster than no_grad for inference
    def _initialize_models(self):
        """Initialize and optimize TTS and tone color converter models."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Initialize models
            self.tone_color_converter = ToneColorConverter(
                'checkpoints_v2/converter/config.json',
                device=self.device
            )
            self.tone_color_converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')
            self.model = TTS(language='EN', device=self.device)

            # Enable TorchScript JIT compilation for faster inference
            if hasattr(torch, 'compile'):
                self.tone_color_converter.model = torch.compile(
                    self.tone_color_converter.model,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                self.model.model = torch.compile(
                    self.model.model,
                    mode="reduce-overhead",
                    fullgraph=True
                )
            
            # Move models to GPU and set to eval mode
            self.tone_color_converter.model.eval()
            self.model.model.eval()

    @torch.inference_mode()
    def generate_speech(self, text: str, reference_speaker: str, speed: float = 1.0) -> str:
        """Generate speech from input text with voice conversion."""
        # Get target speaker embedding
        target_se = se_extractor.get_se(
            reference_speaker,
            self.tone_color_converter,
            **self.se_extract_params
        )[0]
        
        # Measure TTS generation time
        tts_start = time.time()
        self.model.tts_to_file(text, speaker_id=0, output_path=self.temp_path, speed=speed)
        tts_time = time.time() - tts_start
        
        # Measure voice conversion time
        vc_start = time.time()
        self.tone_color_converter.convert(
            audio_src_path=self.temp_path,
            src_se=self.source_se,
            tgt_se=target_se,
            output_path=self.output_path,
            message="@MyShell"
        )
        vc_time = time.time() - vc_start
        
        print(f"TTS generation: {tts_time:.2f}s")
        print(f"Voice conversion: {vc_time:.2f}s")
        print(f"Total speech generation: {tts_time + vc_time:.2f}s")
        
        return self.output_path

def main():
    """Main function demonstrating the usage of VoiceGenerator."""
    generator = VoiceGenerator()
    
    sample_text = "In the year 2194, Earth had gone eerily silent, its once vibrant transmissions reduced to a blanket of static across the stars. From her isolated moonbase orbiting Europa, Lira Sol sent out daily pings into the void, a ritual of hope more than protocol. Then, one evening, through the crackle of cosmic noise, a voice emerged-faint, metallic, yet unmistakably human: 'Lira... this is Kairo. I'm on Mars. You're not alone.' Her heart pounded, but something about the voice felt off-too smooth, too precise, as if it weren't entirely real."
    reference_speaker = "resources/example_reference.mp3"
    
    output_path = generator.generate_speech(sample_text, reference_speaker)
    print(f"Generated speech saved to: {output_path}")

if __name__ == "__main__":
    main()
