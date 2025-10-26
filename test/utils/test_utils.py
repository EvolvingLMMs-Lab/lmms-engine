import unittest

from transformers import AutoProcessor

from lmms_engine.utils.train_utils import TrainUtilities


class TestTrainUtilities(unittest.TestCase):
    def test_convert_open_to_hf(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "./data/images/000000033471.jpg"},
                    },
                    {
                        "type": "text",
                        "text": "\nWhat are the colors of the bus in the image?\nAnswer the question with GPT-T-COCO format.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The bus in the image is white and red."}],
            },
        ]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        print(processor.apply_chat_template(hf_messages))


if __name__ == "__main__":
    unittest.main()
