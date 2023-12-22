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

Try one of these LLM's and prompts that help with generating prompts for musicgen:

- [Musicgen Prompt Generator](https://flowgpt.com/p/musicgen-prompt-generator)
- [Tips for generating prompts with LLMs](https://samim.io/p/2023-11-20-a-gpt-prompt-for-it-to-generate-musicgen-prompts-that/)

## Complete Examples

- `"genre: hiphop", The audio quality is recorded at 320kbps and 48kHz. "Retro Vibes" with a old-school boombap jazz melody. A jazzy and soulful chord progression that is creative and suprising. boom-bap drum patterns, catchy hooks. Inspired by the likes of A Tribe Called Quest, Pete Rock, and DJ Premier"`