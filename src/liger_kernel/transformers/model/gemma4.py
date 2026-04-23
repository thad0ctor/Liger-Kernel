from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.utils import logging

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
from liger_kernel.transformers.model.output_classes import LigerGemma4CausalLMOutputWithPast

logger = logging.get_logger(__name__)


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **loss_kwargs,
) -> Union[Tuple, LigerCausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

    Fused-linear-cross-entropy forward for Gemma4ForCausalLM. Mirrors liger's
    gemma3 causal_forward. Gemma 4 31B uses final_logit_softcapping=30.0, so
    the softcap branch is exercised on the non-fused path.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma4ForCausalLM

    >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-31b")  # illustrative slug
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma4 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with "
            "`AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]
    shift_labels = loss_kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        # final_logit_softcapping via getattr: some future Gemma 4 variants may omit the attribute entirely.
        # Align hidden_states (from last decoder layer, possibly on a sharded
        # GPU) with lm_head.weight (on whatever GPU accelerate placed it) so
        # the fused-LCE matmul doesn't raise cross-device RuntimeError.
        lm_head_device = self.lm_head.weight.device
        if kept_hidden_states.device != lm_head_device:
            kept_hidden_states = kept_hidden_states.to(lm_head_device)
            if labels is not None:
                labels = labels.to(lm_head_device)
            if shift_labels is not None:
                shift_labels = shift_labels.to(lm_head_device)
        # HF Trainer >=4.46 passes ``num_items_in_batch`` (and potentially other
        # metadata tensors) as a GPU scalar on hidden_states' device. Under
        # device_map sharding that can be a different GPU than ``lm_head``, so
        # forwarding the kwargs unchanged would land a stale-device tensor in
        # the loss scaling path. Move any tensor kwargs to ``lm_head_device``.
        for k, v in list(loss_kwargs.items()):
            if isinstance(v, torch.Tensor) and v.device != lm_head_device:
                loss_kwargs[k] = v.to(lm_head_device)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            final_logit_softcapping=getattr(self.config, "final_logit_softcapping", None),
            **loss_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **loss_kwargs,
            )

    if not return_dict:
        output_tuple = (logits,) + outputs[1:]
        output_tuple = (loss,) + output_tuple if loss is not None else output_tuple
        output_tuple = output_tuple + (token_accuracy,) if token_accuracy is not None else output_tuple
        output_tuple = output_tuple + (predicted_tokens,) if predicted_tokens is not None else output_tuple
        return output_tuple

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def multimodal_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **lm_kwargs,
) -> Union[tuple, LigerGemma4CausalLMOutputWithPast]:
    r"""
    Fused-linear-cross-entropy forward for ``Gemma4ForConditionalGeneration``.

    Vision-only scope: audio tensors (``input_features`` / ``input_features_mask``)
    are accepted in the signature for API parity with HF but flow straight through
    to ``self.model(...)``. No audio-specific Liger kernels are applied — patching
    assumes ``audio_config=None`` (audio tower is ``None``).

    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in
        ``[0, ..., config.get_text_config().vocab_size]`` or -100 (see ``input_ids`` docstring).

    image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
        2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
    video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
        2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
    mm_token_type_ids (`torch.LongTensor`, *optional*):
        Multimodal token-type ids used by Gemma 4 to distinguish text / image / video tokens.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    >>> model = Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-...")  # illustrative slug
    >>> processor = AutoProcessor.from_pretrained("google/gemma-4-...")
    ```
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **lm_kwargs,
    )

    shift_labels = lm_kwargs.pop("shift_labels", None)
    # Use the attribute form rather than ``outputs[0]`` — Gemma4 model output
    # is a ModelOutput subclass where ``last_hidden_state`` is the canonical
    # first field.
    hidden_states = outputs.last_hidden_state

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None
    if skip_logits and labels is None:
        raise ValueError("skip_logits is True, but labels is None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None)

    text_config = self.config.get_text_config()

    if skip_logits:
        shift_hidden_states = kept_hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:]

        hidden_device = shift_hidden_states.device
        if attention_mask is not None:
            # we use the input attention mask to shift the hidden_states and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -shift_hidden_states.shape[1] :].to(hidden_device)
            shift_hidden_states = shift_hidden_states[shift_attention_mask.to(hidden_device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
        else:
            shift_hidden_states = shift_hidden_states.contiguous()
            shift_labels = shift_labels.contiguous()

        # Flatten hidden state
        shift_hidden_states = shift_hidden_states.view(-1, text_config.hidden_size)
        shift_labels = shift_labels.view(-1).to(hidden_device)

        # Align with lm_head.weight's device. Under accelerate device_map the
        # last decoder layer and lm_head may land on different GPUs; without
        # this move the fused-LCE matmul raises a cross-device RuntimeError.
        lm_head_device = self.lm_head.weight.device
        if shift_hidden_states.device != lm_head_device:
            shift_hidden_states = shift_hidden_states.to(lm_head_device)
            shift_labels = shift_labels.to(lm_head_device)
        # HF Trainer >=4.46 passes ``num_items_in_batch`` (and potentially other
        # metadata tensors) as a GPU scalar on hidden_states' device. Under
        # device_map sharding that can be a different GPU than ``lm_head``, so
        # forwarding the kwargs unchanged would land a stale-device tensor in
        # the loss scaling path. Move any tensor kwargs to ``lm_head_device``.
        for k, v in list(lm_kwargs.items()):
            if isinstance(v, torch.Tensor) and v.device != lm_head_device:
                lm_kwargs[k] = v.to(lm_head_device)

        result = LigerForCausalLMLoss(
            hidden_states=shift_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=shift_labels,
            hidden_size=text_config.hidden_size,
            shift_labels=shift_labels,
            final_logit_softcapping=getattr(text_config, "final_logit_softcapping", None),
            **lm_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)

    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        elif shift_labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = (loss,) + output if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerGemma4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
