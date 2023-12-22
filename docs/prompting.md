# Prompting Ideas

## Things that work generally

**120bpm** - beats per minute. Typical dance songs are 120. Most ballads are 90-100. A real slow song is 70-85. 140+ is the kinda stuff for raves, techno, & dub

**320kbps 48khz** - these ensure quality of the recordings, reducing hiss & usually expanding the sound range. These numbers are just rather high for an MP3 recording but not near the values of a raw audio recording. Don't apply this to sound that is supposed to be LOFI (like Lofi hip hop) because it undoes their intent : 22kbps quality should start sounding like you're listening to a song through a telephone or through a megaphone

## Model specific

Most finetunes have a system prompt that identifies the finetuned style during training.

For these community finetunes you can try the following one's:

- `pharoAIsanders420/musicgen-medium-boombap`: `"genre: hiphop"`
- `pharoAIsanders420/musicgen-stereo-dub`: `"genre: dub"`
- `pharoAIsanders420/musicgen-small-dnb`: `"genre: dnb"`

## Prompt helper references

To craft effective prompts, one strategy 
is to use "LP-MusicCaps" (https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo) 
to describe any song as text and then ask a LLM: 
`"Take the following text, summarize it and then transform it into a descriptive prompts for a text to music AI: [your LP-Music-Caps generated text]"`

Or try one of these LLM's and prompts that help with generating prompts for musicgen:

- [Musicgen Prompt Generator](https://flowgpt.com/p/musicgen-prompt-generator)
- [Tips for generating prompts with LLMs](https://samim.io/p/2023-11-20-a-gpt-prompt-for-it-to-generate-musicgen-prompts-that/)


## Complete Examples

BOOMBAP:
```
"genre: hiphop", 4/4 320kbps 48khz. "Retro Vibes" with a nostalgic and old-school boombap melody that is reminiscent of the golden era of hip-hop. A jazzy and soulful chord progression. laid-back and chilled-out mood. boom-bap drum patterns, catchy hooks, and smooth flows."
```

DUB:
```
"genre: dub", 4/4 320kbps 48khz, Dub reggae track with a groovy bass line, bouncy percussion, electric guitars, a organ melody, horns, steady drums, punchy snare, and shimmering hi-hats. Laid back, groovy, bouncy, passionate and hypnotic."
```

DNB:
```
"genre: dnb", 4/4 320kbps 48khz, Drum & bass song with a groovy dubstep synth bass line, punchy kick and snare hits, shimmering hi hats, synth pad and repetitive synth lead melody. It sounds energetic and exciting, like something you would hear in underground clubs."
```
