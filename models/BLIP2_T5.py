from typing import Optional, Tuple, List, Union

from types import MethodType
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess


def _split(data, full_batch_size, split_size):
    """
    Takes care of three cases:
    1. data is a tensor: e.g. last_hidden_state, pooler_output etc. split them on the batch_size dim
    2. data is a tuple: e.g. hidden_states, attentions etc. Keep the tuple as it is and split each tensor in it and
       return a list of tuples
    3. data is a tuple of tuples, e.g. past_key_values. Keep the tuple as it is and split each tuple in it and
       return a list of tuples of tuples
    (see documentation of ModelOutput)
    """
    if data is None:
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        return [
            data[i : i + split_size]
            for i in range(0, full_batch_size, split_size)
        ]
    elif isinstance(data, tuple):
        # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
        if isinstance(data[0], tuple):
            return [
                tuple(
                    tuple(tensor[i : i + split_size] for tensor in inner_tuple)
                    for inner_tuple in data
                )
                for i in range(0, full_batch_size, split_size)
            ]

        else:
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        raise ValueError(f"Unexpected attribute type: {type(data)}")


def _split_model_inputs(model_input, split_size, full_batch_size):
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # Edge case: if model_input is None, return a list of Nones
    # this happens with Whisper where encoder_outputs is None
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # Infer the class from the object
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        raise ValueError(
            "`split_size` must be smaller or equal to `full_batch_size`"
        )

    # Helper function to split tensors or tuples of tensors

    # Find all the dataclass fields (e.g., last_hidden_state, pooler_output etc.) and split them
    keys = (
        model_input.__dataclass_fields__.keys()
        if hasattr(model_input, "__dataclass_fields__")
        else model_input.keys()
    )
    # We only keep keys that are in the model_input
    keys = [k for k in keys if k in model_input]
    # Here we can have four types of values: tensors, tuples of tensors and booleans, and encoder_outputs which is a
    # ModelOutput object.
    # bool should not be split but replicated for each split
    bool_keys = [k for k in keys if isinstance(model_input[k], bool)]
    non_bool_keys = [
        k
        for k in keys
        if not isinstance(model_input[k], bool) and not k == "encoder_outputs"
    ]

    # we split the tensors and tuples of tensors
    data_split_list = [
        {
            k: _split(model_input[k], full_batch_size, split_size)[i]
            for k in non_bool_keys
        }
        for i in range(full_batch_size // split_size)
    ]
    # bool values are the same and replicated for each split
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs is a ModelOutput object and should be split by its own
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(
            model_input["encoder_outputs"], split_size, full_batch_size
        )
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]}
            for i, data_split in enumerate(data_split_list)
        ]

    # Convert each dictionary in the list to an object of the inferred class
    split_model_inputs = [
        model_output_cls(**data_split, **bool_data)
        for data_split in data_split_list
    ]

    return split_model_inputs


def beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer,
    logits_processor = None,
    stopping_criteria = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    sequential: Optional[bool] = None,
    **model_kwargs,
):
    # init values
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.

    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    if any(
        model_name in self.__class__.__name__.lower()
        for model_name in ["fsmt", "reformer", "bloom", "ctrl", "gpt_bigcode", "transo_xl", "xlnet", "cpm"]
    ):
        raise RuntimeError(
            f"Currently generation for {self.__class__.__name__} is not supported "
            f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
        )

    inputs_per_sub_batches = _split_model_inputs(
        model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
    )
    return [
        self(
            **inputs_per_sub_batch,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        for inputs_per_sub_batch in inputs_per_sub_batches
    ]


def new_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    decoder_head_mask: Optional[torch.FloatTensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    reduction: Optional[str] = "mean",
) -> None:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

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


@torch.no_grad()
def generate(
    self,
    samples,
    use_nucleus_sampling=False,
    num_beams=5,
    max_length=30,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_captions=1,
    temperature=1,
):
    """
    Args:
        samples (dict): A dictionary containing the following keys:
            - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
        num_beams (int): Number of beams for beam search. 1 means no beam search.
        max_length (int): The maximum length of the sequence to be generated.
        min_length (int): The minimum length of the sequence to be generated.
        top_p (float): The cumulative probability for nucleus sampling.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        num_captions (int): Number of captions to be generated for each image.
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    norm_layer_outputs = []
    image = samples["image"]

    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    input_tokens = samples["tokens"]
    encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

    def print_norm_layer_output(module, input, output):
        norm_layer_outputs.append(output)

    with self.maybe_autocast(dtype=torch.bfloat16):
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids) 
        # inputs_embeds = torch.randn_like(inputs_embeds).to(image.device)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        hook = self.t5_model.encoder.final_layer_norm.register_forward_hook(print_norm_layer_output)
        self.t5_model.forward = MethodType(new_forward, self.t5_model)
        self.t5_model.beam_search = MethodType(beam_search, self.t5_model)

        self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        hook.remove()

    assert len(norm_layer_outputs) == 1
    return None, norm_layer_outputs[0]


@torch.no_grad()
def generate_text_only(
    self,
    samples,
    use_nucleus_sampling=False,
    num_beams=5,
    max_length=30,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_captions=1,
    temperature=1,
):
    """
    Args:
        samples (dict): A dictionary containing the following keys:
            - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
        num_beams (int): Number of beams for beam search. 1 means no beam search.
        max_length (int): The maximum length of the sequence to be generated.
        min_length (int): The minimum length of the sequence to be generated.
        top_p (float): The cumulative probability for nucleus sampling.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        num_captions (int): Number of captions to be generated for each image.
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    norm_layer_outputs = []

    input_tokens = samples["tokens"]

    encoder_atts = input_tokens.attention_mask

    def print_norm_layer_output(module, input, output):
        norm_layer_outputs.append(output)

    with self.maybe_autocast(dtype=torch.bfloat16):
        inputs_embeds = self.t5_model.encoder.embed_tokens(
            input_tokens.input_ids
        )

        hook = self.t5_model.encoder.final_layer_norm.register_forward_hook(
            print_norm_layer_output
        )
        self.t5_model.forward = MethodType(new_forward, self.t5_model)
        self.t5_model.beam_search = MethodType(beam_search, self.t5_model)

        self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        hook.remove()

    assert len(norm_layer_outputs) == 1
    return None, norm_layer_outputs[0]


def get_blip2t5_model(device):
    model, _, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xxl",
        is_eval=True,
        device=device,
    )
    model.generate = MethodType(generate, model)
    return model

def get_tokens(prompt, tokenizer, batch_size, device):
    prompt = [prompt] * batch_size
    input_tokens = tokenizer(prompt, padding="longest", return_tensors="pt").to(device)
    return input_tokens

if __name__ == "__main__":
    from types import MethodType
    import torch
    from PIL import Image
    import requests
    from lavis.models import load_model_and_preprocess
    from torch.profiler import profile, record_function, ProfilerActivity

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # load sample image
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xxl",
        is_eval=True,
        device=device,
    )
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    images = image.repeat([512, 1, 1, 1])
    print(images.shape)
    model.generate = MethodType(generate, model)
    prompt = "What is this picture? Answer:"
    with torch.no_grad():
        for i in range(100):
            print("Epoch: ", i)
            output, encoder_hidden_states = model.generate(
                {"image": images, "prompt": prompt}
            )
            print(encoder_hidden_states[0][0][0][0:10])
    # with profile(
    #     activities=[ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     record_shapes=True,
    # ) as prof:
    #     with record_function("model_inference"):
    #         output, encoder_hidden_states = model.generate(
    #             {"image": image, "prompt": prompt}
    #         )
    # print(output)
    print(encoder_hidden_states[0].size())
    print(len(encoder_hidden_states))
    print(encoder_hidden_states[0][0][0][0:10])
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
