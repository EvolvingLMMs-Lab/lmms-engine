from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SFTChatDataText(BaseModel):
    type: Literal["text"]
    text: str


class SFTChatDataURL(BaseModel):
    url: str


class SFTChatDataVideoURL(SFTChatDataURL):
    """Video URL with optional time-window seek.

    ``video_start`` / ``video_end`` (seconds) let a sample reference a sub-clip
    of a longer source mp4 without having to pre-cut it. Both ends are
    inclusive of the start, exclusive of the end. ``None`` for either falls
    back to "use the whole video on that side".
    """

    video_start: Optional[float] = None
    video_end: Optional[float] = None


class SFTChatDataImage(BaseModel):
    type: Literal["image_url"]
    image_url: SFTChatDataURL


class SFTChatDataAudio(BaseModel):
    type: Literal["audio_url"]
    audio_url: SFTChatDataURL


class SFTChatDataVideo(BaseModel):
    type: Literal["video_url"]
    video_url: SFTChatDataVideoURL


# Hf dataset needs field to be the same across columns
class HFDataContent(BaseModel):
    type: Literal["text", "image_url", "audio_url", "video_url"]
    text: str
    image_url: SFTChatDataURL


SFTChatDataContent = Union[SFTChatDataText, SFTChatDataImage, SFTChatDataAudio, SFTChatDataVideo, HFDataContent]


class SFTChatDataMessages(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[SFTChatDataContent]


class SFTChatData(BaseModel):
    messages: List[SFTChatDataMessages]
    id: int


class PreferenceData(BaseModel):
    id: int
    chosen: List[SFTChatDataMessages]
    rejected: List[SFTChatDataMessages]
    prompt: List[SFTChatDataMessages]
