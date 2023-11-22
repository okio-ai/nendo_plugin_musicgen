from pydantic_settings import BaseSettings
from pydantic import Field


class NendoMusicGenConfig(BaseSettings):
    """
    Default settings for the Nendo musicgen plugin.
    """

    model: str = Field("facebook/musicgen-stereo-large")
    use_melody_conditioning: bool = Field(False)
    duration: float = Field(30.0)
    sample_rate: int = Field(32000)
