# Automatic-Romanian-Text-Generation-using-GPT-2

[![GitHub](https://img.shields.io/badge/GitHub-Automatic-Romanian-Text-Generation-using-GPT-2-181717?logo=github)](https://github.com/MariusCristianBuzea/Automatic-Romanian-Text-Generation-using-GPT-2)
![language](https://img.shields.io/badge/language-Romanian%20NLP-0b7285)
![python](https://img.shields.io/badge/python-3.x-3776ab)
![transformers](https://img.shields.io/badge/transformers-Hugging%20Face-ffcc00)

## Overview

Automatic Romanian text generation with GPT-2 and online news evaluation.

This project is part of a Romanian NLP portfolio and focuses on **Named entity recognition**.

## Data and Results

Dataset notes detected in the original documentation:
- 14,756 news items, while 4922 news items in each of the test and validation datasets were presented
- 4 Examples of Romanian generated sentences**

Reported metrics/results:
- BLEURT checkpoint was used – a self-contained folder that contained a regression model which was tested on several languages, but should work for the 100+ languages of multilingual C4 (a cleaned version of Common Crawl’s web crawl corpus), including the Romanian language with 45M of training and 45
- BLEURT-20 was used as checkpoint, being a 32 layers pre-trained transformer model, named RemBERT, which contained 579M parameters fine-tuned on human ratings and synthetic data (~590K sentence pairs) collected during years 2015 and 2019
- BLEU metric values of generated news items (30 values) using RoGPT-2 and MCBGPT-2
- BLEURT (green) and BERTScore (gray) metrics values (30
- BLEU, ROUGE, BLEURT and BERTScore, the MCBGPT-2 and RoGPT-2

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


## Repository Files

The main README in Romanian remains the detailed source of truth for the original examples, API payloads, Docker commands, model notes and training details.
