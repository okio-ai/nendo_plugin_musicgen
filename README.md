# Nendo Plugin Musicgen 

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

---

![Documentation](https://img.shields.io/website/https/nendo.ai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai)](https://twitter.com/okio_ai) [![](https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat)](https://discord.gg/XpkUsjwXTp)

MusicGen: A state-of-the-art controllable text-to-music model (by [Meta Research](https://github.com/facebookresearch/audiocraft))

## Features

- Generate conditional and unconditional music
- Generate outpaintings from a prompt and a `NendoTrack`
- Use a `NendoTrack` as melody conditioning to generate music
- Use custom finetuned musicgen models

## Requirements

Due to a dependency conflict with `pydantic`, this plugin requires the manual installation of okio's `music` fork:

`pip install git+https://github.com/okio-ai/audiocraft`

## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-musicgen`

## Usage

Take a look at a basic usage example below. 
For more detailed information, please refer to the [documentation](https://okio.ai/docs/plugins).

For more advanced examples, check out the examples folder.
or try it in colab:

<a target="_blank" href="https://colab.research.google.com/drive/1krbzz1OqwCXcLWm5JUIa-otas4TeKZCt?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_musicgen"]))

# load track
track = nd.library.add_track(file_path='/path/to/track.mp3')

# run musicgen with custom model
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
generated_collection[0].play()
```

## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)


## License 

Nendo: MIT License

AudioCraft: MIT License

Pretrained models: The weights are released under the CC-BY-NC 4.0 license