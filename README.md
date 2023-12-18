# Infuser
This is an experimental LLM agent that is meant to be modular to enable the easy addition of different prompting techniques and LLM models.

This is a work-in-progress, and the structure of the project may be refactored any time to achieve the goal of modularity better.

It currently supports the following prompting techniques:

1. Basic - no prompting technique; simple query and answer
2. ReAct - https://arxiv.org/abs/2210.03629

It currently supports the following the models:

1. ChatGPT using an API Key
2. LLaMA 2 using HuggingFace

Technically, the HuggingFace interface should support other models, but no other models apart from LLaMA 2 have been tested.

Copy `.env.example` as `.env` and populate the necessary environment variables to make it work.