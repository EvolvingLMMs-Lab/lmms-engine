from typing import List, Literal, Union

from pydantic import BaseModel, Field


class SFTChatDataText(BaseModel):
    type: Literal["text"]
    text: str


class SFTChatDataURL(BaseModel):
    url: str


class SFTChatDataImage(BaseModel):
    type: Literal["image_url"]
    image_url: SFTChatDataURL


class SFTChatDataAudio(BaseModel):
    type: Literal["audio_url"]
    audio_url: SFTChatDataURL


class SFTChatDataVideo(BaseModel):
    type: Literal["video_url"]
    video_url: SFTChatDataURL


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
