import logging
from typing import Iterable, List, Union

import deepspeed
import torch
from transformers import AutoProcessor

from .logging_utils import Logging


class TrainUtilities:
    @staticmethod
    def is_rank_zero():
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def prepare_model():
        pass

    @staticmethod
    def convert_open_to_hf(messages):
        hf_messages = []
        for message in messages:
            new_message = {"role": message["role"], "content": []}
            for content in message["content"]:
                if content["type"] == "image_url":
                    new_message["content"].append({"type": "image"})
                elif content["type"] == "audio_url":
                    new_message["content"].append(
                        {"type": "audio", "audio_url": content["audio_url"]["url"]}
                    )
                elif content["type"] == "video_url":
                    new_message["content"].append(
                        {"type": "video", "video_url": content["video_url"]["url"]}
                    )
                else:
                    new_content = {"type": "text", "text": content["text"]}
                    if "audio_text" in content:
                        new_content["audio_text"] = content["audio_text"]
                    new_message["content"].append(new_content)
            hf_messages.append(new_message)

        return hf_messages

    @staticmethod
    def sanity_check_labels(
        processor: AutoProcessor, input_ids: torch.Tensor, labels: torch.Tensor
    ):
        print(" ======== Inputs ========")
        for o in processor.batch_decode(input_ids):
            print(o)
            break
        print(" ======== Labels ========")
        labels[labels == -100] = 0
        for o in processor.batch_decode(labels):
            print(o)
            break

    @staticmethod
    def get_device_flops(unit="T"):
        def unit_convert(number, level):
            units = ["B", "K", "M", "G", "T", "P"]
            if number <= 0:
                return number
            ptr = 0
            while ptr < len(units) and units[ptr] != level:
                number /= 1000
                ptr += 1
            return number

        device_name = torch.cuda.get_device_name()
        flops = float("inf")  # INF flops for unkown gpu type

        if "MI300X" in device_name:
            flops = 1336e12
        elif "H100" in device_name or "H800" in device_name or "H200" in device_name:
            flops = 989e12
        elif "A100" in device_name or "A800" in device_name:
            flops = 312e12
        elif "L40" in device_name:
            flops = 181.05e12
        elif "L20" in device_name:
            flops = 119.5e12
        elif "H20" in device_name:
            flops = 148e12
        elif "910B" in device_name:
            flops = 354e12
        elif "RTX 3070 Ti" in device_name:
            flops = 21.75e12
        flops_unit = unit_convert(flops, unit)
        return flops_unit

    @staticmethod
    def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
        to_return = {
            k: t
            for k, t in named_params
            if any(key_match in k for key_match in keys_to_match)
        }
        to_return = {
            k: TrainUtilities.maybe_zero_3(v, ignore_status=True).cpu()
            for k, v in to_return.items()
        }
        return to_return

    @staticmethod
    def maybe_zero_3(param, ignore_status=False, name=None):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    Logging.warning(
                        f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                    )
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param

    @staticmethod
    def get_attn_flops(
        num_layers,
        hidden_size,
        num_heads,
        num_key_value_heads,
        seq_len,
        ffn_hidden_size,
        batch_size=1,
    ):
        """Counts flops for the attention part for a model
        Args:
            num_layers: number of decoder layers
            hidden_size: hidden size of the model
            num_heads: number of heads in the model
            num_key_value_heads: number of key/value heads in the model
            ffn_hidden_size: hidden size of the FFN
            seq_len: sequence length of the decoder
            batch_size: batch size
        Returns:
            model_flops: flops in the model (should be independent of the hardware and model implementation)
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        hidden_size_per_head = hidden_size // num_heads
        # In the following we mark the reduced dimension with parentheses
        # decoder
        # self attention
        ## qkv projection
        qkv_proj_flops_fwd = (
            2
            * num_layers
            * batch_size
            * seq_len
            * (hidden_size)
            * num_heads
            * hidden_size_per_head
            + 2
            * num_layers
            * batch_size
            * seq_len
            * (hidden_size)
            * 2
            * num_key_value_heads
            * hidden_size_per_head
        )
        ## qk logits
        qk_logits_flops_fwd = (
            2
            * num_layers
            * batch_size
            * num_heads
            * seq_len
            * (hidden_size_per_head)
            * seq_len
        )
        ## v logits
        v_logits_flops_fwd = (
            2
            * num_layers
            * batch_size
            * num_heads
            * seq_len
            * (seq_len)
            * hidden_size_per_head
        )
        ## attn out
        attn_out_flops_fwd = (
            2
            * num_layers
            * batch_size
            * num_heads
            * seq_len
            * (hidden_size_per_head)
            * hidden_size
        )
        # FF
        ## 1st layer
        ffn_1_flops_fwd = (
            4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
        )
        ## 2nd layer
        ffn_2_flops_fwd = (
            2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
        )

        flops_fwd = (
            qkv_proj_flops_fwd
            + qk_logits_flops_fwd
            + v_logits_flops_fwd
            + attn_out_flops_fwd
            + ffn_1_flops_fwd
            + ffn_2_flops_fwd
        )
        return flops_fwd

    @staticmethod
    def get_decoder_flops(
        num_layers,
        hidden_size,
        num_heads,
        num_key_value_heads,
        vocab_size,
        seq_len,
        ffn_hidden_size,
        batch_size=1,
    ):
        """Counts flops in an decoder-only model
        Args:
            num_layers: number of decoder layers
            hidden_size: hidden size of the model
            num_heads: number of heads in the model
            num_key_value_heads: number of key/value heads in the model
            ffn_hidden_size: hidden size of the FFN
            vocab_size: size of the vocabulary
            seq_len: sequence length of the decoder
            batch_size: batch size
        Returns:
            model_flops: flops in the model (should be independent of the hardware and model implementation)
        """
        decoder_flops_fwd = TrainUtilities.get_attn_flops(
            num_layers,
            hidden_size,
            num_heads,
            num_key_value_heads,
            seq_len,
            ffn_hidden_size,
            batch_size=batch_size,
        )

        # lm head
        lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

        # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
        # both input and weight tensors
        model_flops = 3 * (
            decoder_flops_fwd + lm_head_flops_fwd
        )  # 1 for fwd + 2 for bwd

        return model_flops

    # Copied from https://github.com/baaivision/EVA/blob/master/EVA-CLIP-18B/shinji/eva_clip/factory.py#L168
    @staticmethod
    def load_zero_partitions(
        model,
        state_dict,
        is_deepspeed_zero3_enabled,
        pretrained_model_path,
        ignore_mismatched_sizes=False,
    ):
        """
        adept from pytorch lightning and transformers
        with deepspeed.zero.Init():
            model = MyModel()
        state_dict = torch.load(model_path, map_location="cpu")
        load_zero_partitions(model, prefix="")
        """

        # because zero3 puts placeholders in model params, this context
        # manager gathers (unpartitions) the params of the current layer, then loads from
        # the state dict and then re-partitions them again
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        loaded_keys = list(state_dict.keys())
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape
                    != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (
                            checkpoint_key,
                            state_dict[checkpoint_key].shape,
                            model_state_dict[model_key].shape,
                        )
                    )
                    del state_dict[checkpoint_key]
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
            if is_deepspeed_zero3_enabled:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(
                    list(module.parameters(recurse=False)), modifier_rank=0
                ):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        load(model_to_load, prefix=start_prefix)
        del state_dict
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )
        if len(unexpected_keys) > 0:
            logging.warning(
                f"Some weights of the model checkpoint at {pretrained_model_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logging.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logging.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )
