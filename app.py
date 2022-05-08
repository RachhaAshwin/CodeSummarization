import streamlit as st
import os
import json
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from model import Seq2Seq
from utils import Example, convert_examples_to_features

def inference(data, model, tokenizer):
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler = eval_sampler, batch_size = len(data))
    model.eval()
    p = []
    for batch in eval_dataloader:
        batch = tuple(t.to('cpu') for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids = source_ids, source_mask = source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces = False)
                p.append(text)
    return (p, source_ids.shape[-1])
  
def get_features(examples, tokenizer):
    features = convert_examples_to_features(
        examples, tokenizer, stage="test"
    )
    all_source_ids = torch.tensor(
        [f.source_ids[: 256] for f in features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask[: 256] for f in features], dtype=torch.long
    )
    return TensorDataset(all_source_ids, all_source_mask)
def build_model(model_class, config, tokenizer):
    encoder = model_class(config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=10,
        max_length=128,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
    )

    model.load_state_dict(
        torch.load(
            "pytorch_model.bin",
            map_location=torch.device("cpu"),
        ),
        strict=False,
    )
    return model


config = RobertaConfig.from_pretrained('microsoft/codebert-base')
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", do_lower_case  = False)

@st.cache(allow_output_mutation=True)
def load_model():
    return build_model( model_class = RobertaModel, config = config, tokenizer = tokenizer).to('cpu')
model = load_model()

st.title("Generate Docstring")
sentence = st.text_area('Please Write a Code here for which you want to generate a docstring :', height=30)
button = st.button("Generate Docstring")

with st.spinner("Generating Docstring"):
    if button and sentence:
        example = [Example(source = sentence, target = None)]
        message, length = inference(get_features(example, tokenizer), model, tokenizer)
        st.write(message)


                   
                   

st.text('Author : Ashwin Rachha')
st.text('Author : Anudeep Reddy')
