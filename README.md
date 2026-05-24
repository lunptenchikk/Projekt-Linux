# Hugging Face Transformers

This project uses pre-trained models from **Hugging Face Transformers** to solve three tasks: sentiment analysis, text generation, and image captioning.

## 1) Sentiment Analysis

I classify movie reviews as positive or negative using
`distilbert-base-uncased-finetuned-sst-2-english`.

The script reads test reviews, runs inference with the transformer, maps the output to `1 / -1`, and computes accuracy against the provided labels.

No training is performed — I only use a pre-trained model and run inference.

## 2) Text Generation

I generate text from a given prompt using `distilgpt2`.

The program supports three decoding methods:

* greedy
* temperature sampling
* beam search

This part demonstrates how different decoding strategies affect the generated output.

## 3) Image Captioning

I generate captions for images using
`Salesforce/blip-image-captioning-base`.

The model analyzes the image and produces a short descriptive sentence.
