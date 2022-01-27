import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset

from rich import print


class CommonGenEval(Dataset):
    def __init__(self, split, tokenizer):
        dataset = load_dataset("common_gen")
        self.tokenizer = tokenizer

        dataset = dataset[split]

        self.dataset = {}

        for d in dataset:
            id_ = d["concept_set_idx"]

            if id_ not in self.dataset.keys():
                self.dataset[id_] = (" ".join(d["concepts"]), [])

            self.dataset[id_][1].append(d["target"])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data[0], [data[1]]

    def __len__(self):
        return len(self.dataset)

    def make_collate_fn(self):
        def collate_fn(batch):
            starters, targets = zip(*batch)
            return (
                self.tokenizer(list(starters), padding=True, return_tensors="pt"),
                targets,
            )

        return collate_fn


def get_outputs(model, device, tokenizer, test_dl, do_sample=False, num_beams=None):
    """
    Runs inference, collects results into a list
    """
    states = []
    generated = []
    examples = []

    for (batch_idx, (encoding, targets)) in enumerate(test_dl):
        input_ids, attention_mask = encoding.input_ids.to(
            device
        ), encoding.attention_mask.to(device)
        labels = targets.input_ids.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            do_sample=do_sample,
            num_beams=num_beams,
        )

        states += tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        examples += tokenizer.batch_decode(labels, skip_special_tokens=True)

    return states, generated, examples


def make_collate(tokenizer):
    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    return collate_fn


def write_output(output_dir, epoch, states, generated, examples):
    """
    Write the model output into text files for evaluation.
    """

    for i in range(10):
        print(f"[red]{states[i]} [green]-> [blue]{generated[i]}")

    os.makedirs(output_dir, exist_ok=True)

    state_file = f"{output_dir}/states_{epoch}.txt"
    generated_file = f"{output_dir}/generated_{epoch}.txt"
    examples_file = f"{output_dir}/examples_{epoch}.txt"

    with open(state_file, "w") as f:
        for i in range(len(generated)):
            f.write(states[i] + "\n")

    with open(generated_file, "w") as f:
        for i in range(len(generated)):
            f.write(generated[i] + "\n")

    with open(examples_file, "w") as f:
        for i in range(len(generated)):
            f.write(examples[i] + "\n")

    return (state_file, generated_file, examples_file)


def get_lm_input(
    self,
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    encoder_outputs=None,
    past_key_values=None,
    inputs_embeds=None,
    decoder_inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r""" """
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if (
        labels is not None
        and decoder_input_ids is None
        and decoder_inputs_embeds is None
    ):
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(
                self.decoder.first_device
            )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)

    return sequence_output
