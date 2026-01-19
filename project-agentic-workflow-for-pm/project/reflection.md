# Limitations of debugging LLMs 
1. Output varies depending on model and temperature. 
Due to the non-deterministic nature of LLMs, during debugging, LLM output can vary wildly. 
It is best to focus on giving clear, accurate prompts and ensure the boundaries are set (eg letting the LLM expand its knowledge when the current context is insufficient)

It is also important to always set `temperature=0` so that our LLM outputs are always deterministic. 
