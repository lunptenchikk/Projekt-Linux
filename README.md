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

## Running with Docker

## 1. Build Docker Image
``` docker build -t tzp-f-huggingface ```

## 2. Run the program using Docker
``` docker run --rm tzp-f-huggingface ```

``` docker run --rm tzp-f-huggingface python hf_sentiment.py test_rec oceny_test_rec.out sentiment/out/oceny_student.out ```

``` docker run --rm tzp-f-huggingface python hf_caption.py caption/images caption/out/caption.txt ```

If you want to see the results of running all the scripts not only in powershell, we should add a command, that will save them into our local copy (using for example VS Code). The commands are below:

```docker run --rm -v "${PWD}:/app" tzp-f-huggingface python hf_textgen.py --method temp --temp 0.7 -seed "The " --len 100 --out textgen/out/generated.txt ```

```docker run --rm -v "${PWD}:/app" tzp-f-huggingface python hf_sentiment.py test_rec oceny_test_rec.out sentiment/out/oceny_student.out```

```docker run --rm -v "${PWD}:/app" tzp-f-huggingface python hf_caption.py caption/images caption/out/caption.txt```
