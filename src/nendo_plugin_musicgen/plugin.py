import os
import time
import yaml
import torch
import subprocess
from logging import Logger
from typing import Any, Optional, Literal
from tqdm import tqdm
import json
import audiocraft
from audiocraft.models import MusicGen
from audiocraft.utils import export
from nendo import Nendo, NendoConfig, NendoGeneratePlugin, NendoTrack, NendoCollection

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

    def train(
        self,
        collection: NendoCollection,
        finetune: bool = False,
        prompt: str = "",
        model: str = settings.model,
        epochs: int = settings.epochs,
        updates_per_epoch: int = settings.updates_per_epoch,
        batch_size: int = settings.batch_size,
        lr: float = settings.lr,
        lr_scheduler: str = settings.lr_scheduler,
        warmup: int = settings.warmup_steps,
        optimizer: str = "adamw",
        cfg_p: float = settings.cfg_p,
        output_dir: str = settings.train_output_dir,
    ):
        """Finetune a pretrained model on a collection of tracks.

        Args:
            collection (NendoCollection): Collection of tracks to finetune on.
            model (Optional[str]): Model version to use.
            epochs (Optional[int]): Number of epochs to finetune.
            updates_per_epoch (Optional[int]): Number of updates per epoch.
            batch_size (Optional[int]): Batch size.
            lr (Optional[float]): Learning rate.
            lr_scheduler (Optional[str]): Learning rate scheduler.
            warmup_steps (Optional[int]): Number of warmup steps.
            output_dir (Optional[str]): Output directory for the finetuned model.

        Returns:
            MusicGen: Finetuned model.
        """
        train_len = 0
        max_sample_rate = settings.sample_rate
        os.makedirs(output_dir, exist_ok=True)

        file_content = ""
        for track in tqdm(collection.tracks()):
            classify_data = track.get_plugin_data("nendo_plugin_classify_core")

            if len(classify_data) < 1:
                self.logger.warning(
                    f"""
                No classification data found for track {track.id}!
                For good finetuning results we recommend running `nendo_plugin_classify_core` first.
                Otherwise conditioning the model will be worse.
                """
                )

            def check_get_data(key):
                data = track.get_plugin_data("nendo_plugin_classify_core", key)
                if len(data) < 1:
                    return ""
                return data[0].value

            entry = {
                "key": check_get_data("key"),
                "artist": track.get_meta("artist") or "",
                "sample_rate": track.sr,
                "file_extension": "wav",
                "description": prompt,
                "keywords": "",
                "duration": track.signal.shape[1] / track.sr,
                "bpm": check_get_data("tempo"),
                "genre": check_get_data("genre"),
                "title": track.get_meta("title") or "",
                "name": track.get_meta("name") or "",
                "instrument": check_get_data("instrument"),
                "moods": check_get_data("moods"),
                "path": track.resource.src,
            }
            max_sample_rate = track.sr

            train_len += 1
            file_content += json.dumps(entry) + "\n"
            track.__dict__["signal"] = None

        with open(output_dir + "/data.jsonl", "a") as train_file:
            train_file.write(file_content)

        # replace default config in audiocraft install dir
        pth = os.path.dirname(audiocraft.__file__)
        default_cfg_path = os.path.join(pth, "../config/dset/audio/default.yaml")
        with open(default_cfg_path, "r") as file:
            data = yaml.safe_load(file)

        data["datasource"]["max_sample_rate"] = max_sample_rate
        data["datasource"]["max_channels"] = 2
        data["datasource"]["train"] = output_dir
        data["datasource"]["valid"] = output_dir
        data["datasource"]["evaluate"] = output_dir
        data["datasource"]["generate"] = output_dir

        with open(default_cfg_path, "w") as file:
            file.write("# @package __global__\n")
            yaml.dump(data, file)

        # replace dora default output dir in yaml
        default_cfg_path = os.path.join(pth, "../config/teams/default.yaml")
        with open(default_cfg_path, "r") as file:
            data = yaml.safe_load(file)

        data["default"]["dora_dir"] = output_dir

        with open(default_cfg_path, "w") as file:
            yaml.dump(data, file)

        if model not in ["melody", "stereo-melody"]:
            solver = "musicgen/musicgen_base_32khz"
            model_scale = model.rsplit("-")[-1]
            conditioner = "text2music"
        else:
            solver = "musicgen/musicgen_melody_32khz"
            model_scale = "medium"
            conditioner = "chroma2music"
        continue_from = f"//pretrained/{model}"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        args = [
            "python",
            "-m",
            "dora",
            "-P",
            "audiocraft",
            "run",
            f"solver={solver}",
            f"model/lm/model_scale={model_scale}",
            f"conditioner={conditioner}",
            f"device={device}",
        ]
        if finetune:
            args.append(f"continue_from={continue_from}")
        if "stereo" in model:
            args.append(f"codebooks_pattern.delay.delays={[0, 0, 1, 1, 2, 2, 3, 3]}")
            args.append("transformer_lm.n_q=8")
            args.append("interleave_stereo_codebooks.use=True")
            args.append("channels=2")

        args.append(f"datasource.max_sample_rate={max_sample_rate}")
        args.append(f"datasource.train={output_dir}")

        # for quicker training we ignore validation and generation
        args.append(f"dataset.valid.num_samples={1}")
        args.append(f"dataset.generate.num_samples={1}")
        args.append(f"dataset.train.num_samples={len(collection)}")

        args.append(f"optim.epochs={epochs}")
        args.append(f"optim.lr={lr}")
        args.append(f"schedule.lr_scheduler={lr_scheduler}")
        args.append(f"schedule.cosine.warmup={warmup}")
        args.append(f"schedule.polynomial_decay.warmup={warmup}")
        args.append(f"schedule.inverse_sqrt.warmup={warmup}")
        args.append(f"schedule.linear_warmup.warmup={warmup}")
        args.append(f"classifier_free_guidance.training_dropout={cfg_p}")

        if updates_per_epoch is not None:
            args.append(
                f"logging.log_updates={updates_per_epoch // 10 if updates_per_epoch // 10 >= 1 else 1}"
            )
        else:
            args.append(f"logging.log_updates=0")
        args.append(f"dataset.batch_size={batch_size}")
        args.append(f"optim.optimizer={optimizer}")

        if updates_per_epoch is None:
            args.append("dataset.train.permutation_on_files=False")
            args.append("optim.updates_per_epoch=1")
        else:
            args.append("dataset.train.permutation_on_files=True")
            args.append(f"optim.updates_per_epoch={updates_per_epoch}")

        # call dora subprocess
        self.logger.warning("Starting training with args:\n" + " ".join(args))
        subprocess.check_output(args)

        # export model after training
        # get model checkpoint path
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in [f for f in filenames if f == "checkpoint.th"]:
                checkpoint_dir = os.path.join(dirpath, filename)

        print("Exporting model from checkpoint dir:", checkpoint_dir)

        export.export_lm(checkpoint_dir, os.path.join(output_dir, "state_dict.bin"))
        export.export_pretrained_compression_model(
            "facebook/encodec_32khz",
            os.path.join(output_dir, "compression_state_dict.bin"),
        )
        print("Run your new model using the nendo_plugin_musicgen plugin:")
        print("from nendo import Nendo")
        print("nd = Nendo(plugins=['nendo_plugin_musicgen'])")
        print("generations = nd.plugins.musicgen(model='" + output_dir + "')")
        return

    @NendoGeneratePlugin.run_track
    def generate(
        self,
        track: Optional[NendoTrack] = None,
        bpm: int = 0,
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

        if (
            track is not None
            and len(track.get_plugin_data("nendo_plugin_classify_core")) > 1
        ):
            bpm = int(
                float(track.get_plugin_value("tempo")),
            )
            key = track.get_plugin_value("key")
            scale = track.get_plugin_value("scale")

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
            logger=self.logger,
            global_prompt=prompt,
            temperature=temperature,
            bpm=bpm,
            key=key,
            scale=scale,
            sr_select=sr or settings.sample_rate,
            trim_start=start_time,
            trim_end=(
                duration
                if use_melody_conditioning
                else (duration or settings.duration) - conditioning_length
            ),
            overlay=conditioning_length,
            duration=duration,
            sample=musicgen_sample if not use_melody_conditioning else None,
            melodies=[musicgen_sample] if use_melody_conditioning else None,
            cfg_coef=cfg_coef,
            seed=seed,
            n_samples=n_samples,
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
