from logging import Logger
from typing import Any, Optional, Literal

from audiocraft.models import MusicGen
from nendo import (
    Nendo,
    NendoConfig,
    NendoGeneratePlugin,
    NendoTrack,
)

from .config import NendoMusicGenConfig
from .musicgen import load_model, do_predictions

settings = NendoMusicGenConfig()


class NendoMusicGen(NendoGeneratePlugin):
    """A nendo plugin for music generation based on MusicGen by Facebook AI Research.
    https://github.com/facebookresearch/audiocraft/

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nendo = Nendo(config=NendoConfig(plugins=["nendo_plugin_musicgen"]))
        track = nendo.library.add_track_from_file(
            file_path="path/to/file.wav",
        )

        outpaintings = nendo.plugins.musicgen(
            track=track,
            prompt="thrash metal, distorted guitars",
            bpm=120,
            key="D",
            scale="Minor",
            n_samples=5
        )

        outpaintings.tracks()[0].play()
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    logger: Logger = None
    model: MusicGen = None
    current_version: str = None

    def __init__(self, **data: Any):
        """Initialize plugin."""
        super().__init__(**data)
        self.logger.info("Initializing plugin: MUSICGEN")

    @NendoGeneratePlugin.run_track
    def run_track(
            self,
            track: Optional[NendoTrack] = None,
            bpm: int = 120,
            key: Literal[
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "Bb",
                "B",
            ] = "",
            scale: Literal["Major", "Minor"] = "",
            prompt: str = "",
            temperature: float = 1.0,
            cfg_coef: float = 3.5,
            start_time: int = 0,
            duration: int = 30,
            conditioning_length: int = 1,
            seed: int = -1,
            n_samples: int = 2,
            model: str = settings.model,
            use_melody_conditioning: bool = settings.use_melody_conditioning,
    ):
        """Generate an outpainting from a track, use a track as melody conditioning or generate a track from scratch.

        Args:
            track (Optional[NendoTrack]): Track to generate from, will be used
                as melody conditioning if `use_melody_conditioning` is `True`.
            bpm (Optional[int]): Beats per minute of the generated track.
            key (Optional[Literal["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]]):
                Key of the generated track.
            scale (Optional[Literal["Major", "Minor"]]): Scale of the generated track.
            prompt (Optional[str]): Prompt for the generation.
            temperature (Optional[float]): Temperature for the generation. Controls how "random" the next token will be.
            cfg_coef (Optional[float]): Coefficient for the generation. Controls how strong the prompt is used
                as a conditioning signal.
            start_time (Optional[int]): Start time for the generation.
            duration (Optional[int]): Duration of the generation.
            conditioning_length (Optional[int]): Conditioning length for the generation.
            seed (Optional[int]): Seed for the generation.
            n_samples (Optional[int]): Number of samples to generate.
            model (Optional[str]): Model version to use.
            use_melody_conditioning (Optional[bool]): Whether to use melody conditioning.

        Returns:
            List[NendoTrack]: List of generated tracks.
        """
        if use_melody_conditioning:
            assert (
                    "melody" in model or "melody" in self.current_version
            ), "Melody conditioning needs a model trained on melody conditioning."

        if model != self.current_version:
            self.model = load_model(version=model)
            self.current_version = model

        if track is None:
            musicgen_sample = None
            sr = None
        else:
            y, sr = track.signal, track.sr
            musicgen_sample = (sr, y[:, : sr * duration].T)

        if track is not None and len(track.get_plugin_data("nendo_plugin_classify_core")) > 1:
            bpm = int(float(track.get_plugin_data("nendo_plugin_classify_core", "tempo")))
            key = track.get_plugin_data("nendo_plugin_classify_core", "key")
            scale = track.get_plugin_data("nendo_plugin_classify_core", "scale")

        params = {
            "prompt": prompt,
            "temperature": temperature,
            "cfg_coef": cfg_coef,
            "start_time": start_time,
            "duration": duration,
            "conditioning_length": conditioning_length,
            "seed": seed,
            "model_version": model,
        }

        outputs = do_predictions(
            model=self.model,
            global_prompt=prompt,
            temperature=temperature,
            bpm=bpm,
            key=key,
            scale=scale,
            sr_select=sr or settings.sample_rate,
            trim_start=start_time,
            trim_end=duration
            if use_melody_conditioning
            else (duration or settings.duration) - conditioning_length,
            overlay=conditioning_length,
            duration=duration,
            sample=musicgen_sample if not use_melody_conditioning else None,
            melodies=[musicgen_sample] if use_melody_conditioning else None,
            cfg_coef=cfg_coef,
            seed=seed,
            n_samples=1 if use_melody_conditioning else n_samples,
        )

        if track is None:
            return [
                self.nendo_instance.library.add_track_from_signal(
                    signal=output,
                    sr=settings.sample_rate,
                    meta={"generation_parameters": params or {}},
                )
                for output in outputs
            ]
        return [
            self.nendo_instance.library.add_related_track_from_signal(
                signal=output,
                sr=sr,
                track_type="musicgen",
                relationship_type="musicgen",
                related_track_id=track.id,
                track_meta={"generation_parameters": params or {}},
            )
            for output in outputs
        ]
