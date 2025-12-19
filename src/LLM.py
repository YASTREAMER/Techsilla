import re
import gc
import nltk
import torch
from time import sleep
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def QuestionGeneration(model, tokenizer, AudioText) -> list:

    messages = [
        {
            "role": "system",
            "content": "You are a tech interviewer and you only ask them question and do not have to say anything else and you have to ask the candidate question based on what they specialize in. You only have to ask them about tech interview question such as DSA or ML related question. If they give you information about any project that they have done then you will have to ask them question regarding that project. Ask the canditate around 10 questions",
        },
        {"role": "user", "content": AudioText},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    content = content + "\n"

    questions = re.findall("(.+?)\\n", content)

    return questions


def QuestionAnswer(model, tokenizer, Answer, Category) -> None:

    messages = [
        {
            "role": "system",
            "content": f"Your role right now is to check the answer given by the canditate and judge it's accuracy. The questions asked are based on projects and {Category}",
        },
        {"role": "user", "content": Answer},
    ]
    print(messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    content = content + "\n"

    print("thinking content:", thinking_content)
    print("content:", content)


def AudioToText(device):
    model_id = "distil-whisper/distil-large-v3.5"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        dtype=torch_dtype,
        device=device,
    )

    result = pipe("output.wav")
    return result


def runner(Text: str) -> list:
    model_name = "Qwen/Qwen3-0.6B"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    )

    Questions = QuestionGeneration(model, tokenizer, Text)
    modelDel(model)
    modelDel(tokenizer)
    return Questions


def grammar_score(sentence, corrector):
    corrected = corrector(sentence, max_length=128)[0]["generated_text"]
    # Simple token-based edit distance
    dist = nltk.edit_distance(sentence.split(), corrected.split())
    max_len = max(len(sentence.split()), len(corrected.split()))
    score = 1 - dist / max_len  # normalized accuracy score
    return corrected, round(score * 100, 2)


def modelDel(model):

    del model
    # Run garbage collection
    gc.collect()

    # Clear CUDA cache if running on GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
