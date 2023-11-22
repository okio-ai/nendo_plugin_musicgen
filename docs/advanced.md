# Advanced Usage

## Unconditional Generation

The basic mode of usage for `nendo-plugin-musicgen` is to generate unconditional music.
If no `NendoTrack` or `NendoCollection` is given when calling the plugin,
it will generate music from scratch.

```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_musicgen"]))
generated_collection = nd.plugins.musicgen(
    n_samples=5,
    prompt="janelle monae, rnb, funky, fast, futuristic",
    bpm=116,
    key="C",
    scale="Major",
    duration=30
)
```

## Outpainting

Outpainting is the process of generating music from a prompt and a `NendoTrack`.
The plugin will use the `NendoTrack`'s signal as a conditioning signal for the music generation
and continue the track from there.

!!! note
    A very important parameter for this mode is `conditioning_length`.
    This parameter determines how many seconds of the `NendoTrack`'s
    signal will be used for conditioning and when the outpainting starts.

```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_musicgen"]))
track = nd.library.add_track(file_path='/path/to/track.mp3')

generated_collection = nd.plugins.musicgen(
    track=track,
    n_samples=5,
    prompt="janelle monae, rnb, funky, fast, futuristic",
    bpm=116,
    key="C",
    scale="Major",
    duration=30,
    conditioning_length=10
)
```

## Melody Conditioning

Melody conditioning is the process of generating music from a prompt and a `NendoTrack`, where the given track's
melody is used as a conditioning signal for the music generation.

Compared to outpainting, this mode will not continue the track but instead
create a new track based on the given track's melody.

!!! warning
    Also be sure to use a model that supports melody conditioning.

```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_musicgen"]))
track = nd.library.add_track(file_path='/path/to/track.mp3')

generated_collection = nd.plugins.musicgen(
    track=track,
    n_samples=5,
    prompt="janelle monae, rnb, funky, fast, futuristic",
    bpm=116,
    key="C",
    scale="Major",
    duration=30,
    use_melody_conditioning=True,
    model="facebook/musicgen-melody"
)
```

## Parameters

| Parameter               | Description                                                                                           | Default Value     |
|-------------------------|-------------------------------------------------------------------------------------------------------|-------------------|
| bpm                     | Beats per minute of the generated track.                                                              | 120               |
| key                     | Key of the generated track. Choices: "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B". | "C"               |
| scale                   | Scale of the generated track, either "Major" or "Minor".                                              | "Major"           |
| prompt                  | Prompt for the generation.                                                                            | "" (empty string) |
| temperature             | Temperature for the generation, controlling randomness of the next token.                             | 1.0               |
| cfg_coef                | Coefficient for the generation, influencing the strength of the prompt as a conditioning signal.      | 3.5               |
| start_time              | Start time for the generation.                                                                        | 0                 |
| duration                | Duration of the generation in seconds.                                                                | 30                |
| conditioning_length     | Conditioning length for the generation in seconds.                                                    | 6                 |
| seed                    | Seed for the generation.                                                                              | -1                |
| n_samples               | Number of samples to generate.                                                                        | 1                 |
| use_melody_conditioning | Whether to use melody conditioning in the generation process.                                         | False             |

