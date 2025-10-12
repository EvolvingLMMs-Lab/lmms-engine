import os
from typing import Dict

import torch

from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities

from .vision_audio_dataset import MAX_AUDIO_LENGTH, VisionAudioSFTDataset


@register_dataset("qwen_omni")
class QwenOmniSFTDataset(VisionAudioSFTDataset):
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
                    audio_url = content["audio_url"]["url"]
                    # Skip placeholders from video extraction, they handled by video processing
                    if audio_url == "from_video":
                        continue

                    loaded_audios = self.load_audio(
                        audio_url,
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
                    video_url = content["video_url"]["url"]
                    if data_folder is not None:
                        video_path = os.path.join(data_folder, video_url)
                    else:
                        video_path = video_url
                    frames, sample_fps = self.load_videos(
                        video_url,
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps

                    # check if audio was extracted from video
                    if (
                        hasattr(self, "video_extracted_audio")
                        and video_path in self.video_extracted_audio
                    ):
                        extracted_audio = self.video_extracted_audio[video_path]
                        kwargs["use_audio_in_video"] = True

                        if hasattr(self.processor, "sampling_rate"):
                            max_audio_samples = (
                                MAX_AUDIO_LENGTH * self.processor.sampling_rate
                            )
                            # minimum audio length (2 seconds) to avoid pooling errors
                            min_audio_samples = 2 * self.processor.sampling_rate

                            audio_splits = []
                            for i in range(0, len(extracted_audio), max_audio_samples):
                                audio_chunk = extracted_audio[i : i + max_audio_samples]
                                if len(audio_chunk) >= min_audio_samples:
                                    audio_splits.append(audio_chunk)

                            audios.extend(audio_splits)

                            # audio placeholders to content if audio was extracted for processor compatibility
                            for _ in range(len(audio_splits)):
                                new_content.append(
                                    {
                                        "type": "audio_url",
                                        "audio_url": {"url": "from_video"},
                                    }
                                )
                        else:
                            audios.append(extracted_audio)
                            new_content.append(
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": "from_video"},
                                }
                            )
                        del self.video_extracted_audio[video_path]
                    else:
                        kwargs["use_audio_in_video"] = False

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
