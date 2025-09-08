from typing import Dict

import torch

from lmms_engine.utils.train_utils import TrainUtilities

from .vision_dataset import VisionSFTDataset

MAX_AUDIO_LENGTH = 30

from lmms_engine.mapping_func import register_dataset


@register_dataset("vision_audio")
class VisionAudioSFTDataset(VisionSFTDataset):
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images = []
        audios = []
        videos = []
        messages = data["messages"]
        new_messages = []
        kwargs = {}
        for message in messages:
            new_content = []
            for idx, content in enumerate(message["content"]):
                if content["type"] == "image_url":
                    images.append(
                        self.load_image(
                            content["image_url"]["url"], data_folder=data_folder
                        )
                    )
                    new_content.append(content)
                elif content["type"] == "audio_url":
                    loaded_audios = self.load_audio(
                        content["audio_url"]["url"],
                        sr=self.processor.sampling_rate,
                        data_folder=data_folder,
                    )
                    audio_splits = []
                    # Split the loaded audio to 30s chunks and extend the messages content
                    for i in range(
                        0,
                        len(loaded_audios),
                        MAX_AUDIO_LENGTH * self.processor.sampling_rate,
                    ):
                        audio_splits.append(
                            loaded_audios[
                                i : i + MAX_AUDIO_LENGTH * self.processor.sampling_rate
                            ]
                        )
                    for _ in range(len(audio_splits)):
                        new_content.append(content)
                    audios.extend(audio_splits)
                elif content["type"] == "video_url":
                    # Loading videos with fps
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs
                    kwargs["fps"] = sample_fps
                    new_content.append(content)
                else:
                    new_content.append(content)
            message["content"] = new_content
            new_messages.append(message)
        messages = new_messages

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            videos=videos,
            sampling_rate=self.processor.sampling_rate,
            **kwargs,
        )
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        return super().load_from_hf(data)
