import random
from typing import Literal, Tuple, Optional, List

import numpy as np
import torch
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MusicGen


def load_model(
    version: str,
) -> MusicGen:
    """Load a pretrained model from the audiocraft library.

    Args:
        version (str): The version of the model to load.

    Returns:
        MusicGen: The pretrained model.
    """
    model = MusicGen.get_pretrained(version)
    model.lm.to("cuda")
    return model


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to [0, 1].

    Args:
        audio_data (np.ndarray): The audio data to normalize.

    Returns:
        np.ndarray: The normalized audio data.
    """
    audio_data = audio_data.astype(np.float32)
    max_value = np.max(np.abs(audio_data))
    audio_data /= max_value
    return audio_data


def do_predictions(
    model: MusicGen,
    global_prompt: str,
    temperature: float,
    bpm: int,
    key: Literal["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
    scale: Literal["Major", "Minor"],
    trim_start: int,
    trim_end: int,
    overlay: int,
    duration: int,
    sample: Optional[Tuple[int, np.ndarray]] = None,
    melodies: Optional[List[Tuple[int, np.ndarray]]] = None,
    sr_select: int = 44100,
    progress: Optional[bool] = False,
    cfg_coef: Optional[float] = 3.5,
    seed: Optional[int] = -1,
    n_samples: Optional[int] = 1,
    **gen_kwargs,
) -> List[np.ndarray]:
    """Generate audio from a pretrained model.

    Args:
        model (MusicGen): The pretrained model.
        global_prompt (str): The prompt for the generation.
        temperature (float): The temperature for the generation. Controls how "random" the next token will be.
        bpm (int): The bpm of the generated track.
        key (Literal["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]): The key of the generated track.
        scale (Literal["Major", "Minor"]): The scale of the generated track.
        trim_start (int): The start time for the generation.
        trim_end (int): The end time for the generation.
        overlay (int): The overlay for the generation.
        duration (int): The duration of the generation.
        sample (Optional[Tuple[int, np.ndarray]]): The sample to use for the generation.
        melodies (Optional[List[Tuple[int, np.ndarray]]]): The melodies to use for the generation.
        channel (Optional[Literal["stereo", "mono"]]): The channel to use for the generation.
        sr_select (Optional[str]): The sample rate to use for the generation.
        progress (Optional[bool]): Whether to show a progress bar for the generation.
        cfg_coef (Optional[float]): The coefficient for the generation. Controls how strong the prompt is used as a
            conditioning signal.
        seed (Optional[int]): The seed for the generation.
        n_samples (Optional[int]): The number of samples to generate.
        **gen_kwargs: Additional generation arguments.

    Returns:
        List[np.ndarray]: The generated audio.
    """
    if trim_start < 0:
        trim_start = 0
    if trim_end < 0:
        trim_end = 0

    seed = random.randint(0, 0xFFFF_FFFF_FFFF) if seed == -1 else seed
    torch.manual_seed(seed)
    maximum_size = 29.5
    cut_size = 0
    sample_p = None

    if sample is not None:
        global_sr, sample_m = sample[0], sample[1]
        sample_m = normalize_audio(sample_m)
        sample_m = torch.from_numpy(sample_m).t()
        if sample_m.dim() == 1:
            sample_m = sample_m.unsqueeze(0)
        sample_length = sample_m.shape[sample_m.dim() - 1] / global_sr
        if trim_start >= sample_length:
            trim_start = sample_length - 0.5
        if trim_end >= sample_length:
            trim_end = sample_length - 0.5
        if trim_start + trim_end >= sample_length:
            tmp = sample_length - 0.5
            trim_start = tmp / 2
            trim_end = tmp / 2
        sample_m = sample_m[
            ...,
            int(global_sr * trim_start) : int(global_sr * (sample_length - trim_end)),
        ]
        sample_length = sample_length - (trim_start + trim_end)
        if sample_length > maximum_size:
            cut_size = sample_length - maximum_size
            sample_p = sample_m[..., : int(global_sr * cut_size)]
            sample_m = sample_m[..., int(global_sr * cut_size) :]
        if sample_length >= duration:
            duration = sample_length + 0.5

    texts = []
    bpm_str = str(bpm) + " bpm"
    if key != "" and scale != "":
        key_str = ", " + str(key) + " " + str(scale)
    else:
        key_str = ""
    global_str = (", " + str(global_prompt)) if str(global_prompt) != "" else ""
    texts.append(bpm_str + key_str + global_str)

    text_single = texts[0]
    texts = [text_single for _ in range(n_samples)]

    model.set_generation_params(
        temperature=temperature,
        duration=(duration - cut_size),
        extend_stride=model.max_duration - overlay,
        cfg_coef=cfg_coef,
        **gen_kwargs,
    )

    print(
        "new batch",
        len(texts),
        texts,
        "melodies:",
        [None if m is None else (m[0], m[1].shape) for m in melodies]
        if melodies
        else None,
        "sample:",
        [None if sample is None else (sample[0], sample[1].shape)] if sample else None,
    )

    processed_melodies = []
    target_sr = 32000
    target_ac = 1

    if melodies is not None:
        for melody in melodies:
            if melody is None:
                processed_melodies.append(None)
            else:
                sr, melody = (
                    melody[0],
                    torch.from_numpy(melody[1]).to(model.device).float().t(),
                )
                if melody.dim() == 1:
                    melody = melody[None]
                melody = melody[..., : int(sr * duration)]
                melody = convert_audio(melody, sr, target_sr, target_ac)
                processed_melodies.append(melody)

    if sample is not None:
        if sample_p is None:
            # expand to repeat the sample n_sample times
            sample_m = sample_m.unsqueeze(0).repeat(
                n_samples, *[1 for _ in range(sample_m.dim())]
            )

            outputs = model.generate_continuation(
                prompt=sample_m,
                prompt_sample_rate=global_sr,
                descriptions=texts,
                progress=progress,
            )
        else:
            if sample_p.dim() > 1:
                sample_p = convert_audio(sample_p, global_sr, target_sr, target_ac)
            sample_p = sample_p.to(model.device).float().unsqueeze(0)

            # expand to repeat the sample n_sample times
            sample_m = sample_m.unsqueeze(0).repeat(
                n_samples, *[1 for _ in range(sample_m.dim())]
            )
            sample_p = sample_p.unsqueeze(0).repeat(
                n_samples, *[1 for _ in range(sample_p.dim())]
            )

            outputs = model.generate_continuation(
                prompt=sample_m,
                prompt_sample_rate=global_sr,
                descriptions=texts,
                progress=progress,
            )
            outputs = torch.cat([sample_p, outputs], 2)

    elif any(m is not None for m in processed_melodies):
        outputs = model.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
        )
    else:
        outputs = model.generate(texts, progress=progress)

    outputs = outputs.detach().cpu().float()
    outputs = convert_audio(outputs, target_sr, sr_select, 2)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return outputs
