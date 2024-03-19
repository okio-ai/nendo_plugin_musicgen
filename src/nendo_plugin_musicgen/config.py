from pydantic_settings import BaseSettings
from pydantic import Field


class NendoMusicGenConfig(BaseSettings):
    """
    Default settings for the Nendo musicgen plugin.
    """

    model: str = Field("facebook/musicgen-stereo-medium")
    use_melody_conditioning: bool = Field(False)
    duration: float = Field(30.0)
    sample_rate: int = Field(32000)
    epochs: int = Field(1)
    updates_per_epoch: int = Field(100)
    batch_size: int = Field(1)
    lr: float = Field(0.1)
    lr_scheduler: str = Field("cosine")
    warmup_steps: int = Field(8)
    train_output_dir: str = Field("train_output")
    cfg_p: float = Field(0.3)
