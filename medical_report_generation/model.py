from typing import cast

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertTokenizer,
    CLIPProcessor,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    DistilBertModel,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    ImageToTextPipeline,
    LogitsProcessor,
    LogitsProcessorList,
    TextGenerationPipeline,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

VIT_PATH = "openai/clip-vit-large-patch14"
GPT2_PATH = "openai-community/gpt2"
VISION_ENCODER_DECODER_PATH = "nlpconnect/vit-gpt2-image-captioning"


# class MedicalReportGeneration(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         gpt = GPT2LMHeadModel.from_pretrained(GPT2_PATH)
#         gpt = cast(GPT2LMHeadModel, gpt)
#         self.gpt = gpt
#         tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_PATH)
#         tokenizer = cast(GPT2TokenizerFast, tokenizer)
#         self.tokenizer = tokenizer
#         clip_processor = CLIPProcessor.from_pretrained(VIT_PATH)
#         clip_processor = cast(CLIPProcessor, clip_processor)
#         self.clip_processor = clip_processor
#         image_processor = ViTImageProcessor.from_pretrained(VISION_ENCODER_DECODER_PATH)
#         image_processor = cast(ViTImageProcessor, image_processor)
#         self.image_processor = image_processor
#         image_encoder_projection = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)
#         image_encoder_projection = cast(CLIPVisionModelWithProjection, image_encoder_projection)
#         self.image_encoder = image_encoder_projection

#     def forward(self, images: torch.Tensor, label: str):
#         pixel_values = self.clip_processor(images=images, return_tensors="pt").data["pixel_values"]
#         labels = self.tokenizer(label, return_tensors="pt")
#         # image embeds (1,768)
#         encoder_output = self.image_encoder(pixel_values)
#         # GPT LM Head: logits (1,50257)
#         # logits = self.gpt(inputs_embeds=encoder_output.image_embeds).logits
#         # loss = CrossEntropyLoss()(
#         #     logits.reshape(-1, self.gpt.config.vocab_size), labels.input_ids.reshape(-1)
#         # )
#         # GPT: last hidden state (1,768)
#         # output = self.gpt(inputs_embeds=encoder_output.image_embeds)
#         # generated_ids = self.gpt.generate(
#         #     self.image_processor(images).pixel_values[0],
#         #     max_length=16,
#         #     num_beams=4,
#         #     return_dict_in_generate=True,
#         # )
#         # generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         ...


class MedicalReportGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # load a fine-tuned image captioning model and corresponding tokenizer and image processor
        encoder_decoder = VisionEncoderDecoderModel.from_pretrained(VISION_ENCODER_DECODER_PATH)
        encoder_decoder = cast(VisionEncoderDecoderModel, encoder_decoder)
        self.encoder_decoder = encoder_decoder
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tokenizer = cast(BertTokenizer, tokenizer)
        self.tokenizer = tokenizer
        encoder_decoder.config.decoder_start_token_id = tokenizer.cls_token_id
        encoder_decoder.config.pad_token_id = tokenizer.pad_token_id
        image_processor = ViTImageProcessor.from_pretrained(VISION_ENCODER_DECODER_PATH)
        image_processor = cast(ViTImageProcessor, image_processor)
        self.image_processor = image_processor

    def forward(self, images: torch.Tensor, label_str: str):
        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
        # autoregressively generate caption (uses greedy decoding by default)
        # (1,16) tokens
        # generated_ids = self.encoder_decoder.generate(pixel_values)
        labels = self.tokenizer(label_str, return_tensors="pt").input_ids
        output = self.encoder_decoder(pixel_values=pixel_values, labels=labels)
        # generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(generated_text)
        return output
