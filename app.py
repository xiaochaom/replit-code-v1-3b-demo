"""Inspired by the SantaCoder demo Huggingface space. 
Link: https://huggingface.co/spaces/bigcode/santacoder-demo/tree/main/app.py
"""

import os
import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

REPO = "replit/replit-code-v1-3b"

description = """# <h1 style="text-align: center; color: white;"><span style='color: #F26207;'> Code Completion with replit-code-v1-3b </h1>
<span style="color: white; text-align: center;"> replit-code-v1-3b model is a 2.7B LLM trained on 20 languages from the Stack Dedup v1.2 dataset. You can click the button several times to keep completing your code.</span>"""


token = os.environ["HUB_TOKEN"]
device = "cuda" if torch.cuda.is_available() else "cpu"

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"
MAX_INPUT_TOKENS = 1024 # max tokens from context


tokenizer = AutoTokenizer.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True)
tokenizer.truncation_side = "left" # ensures if truncate, then keep the last N tokens of the prompt going L -> R

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True, low_cpu_mem_usage=True).to(device, dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True, low_cpu_mem_usage=True)

model.eval()


custom_css = """
.gradio-container {
    background-color: #0D1525; 
    color:white
}
#orange-button {
    background: #F26207 !important;
    color: white;
}
.cm-gutters{
    border: none !important;
}
"""

def post_processing(prompt, completion):
    return prompt + completion
    # completion = "<span style='color: #499cd5;'>" + completion + "</span>"
    # prompt = "<span style='color: black;'>" + prompt + "</span>"
    # code_html = f"<hr><br><pre style='font-size: 14px'><code>{prompt}{completion}</code></pre><br><hr>"
    # return code_html

    
def code_generation(prompt, max_new_tokens, temperature=0.2, seed=42, top_p=0.9, top_k=None, use_cache=True, repetition_penalty=1.0):

    # truncates the prompt to MAX_INPUT_TOKENS if its too long
    x = tokenizer.encode(prompt, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True).to(device)
    print("Prompt shape: ", x.shape) # just adding to see in the space logs in prod
    set_seed(seed)
    y = model.generate(x, 
                       max_new_tokens=max_new_tokens, 
                       temperature=temperature, 
                       pad_token_id=tokenizer.pad_token_id, 
                       eos_token_id=tokenizer.eos_token_id, 
                       top_p=top_p,
                       top_k=top_k,
                       use_cache=use_cache,
                       repetition_penalty=repetition_penalty
                    )
    completion = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    completion = completion[len(prompt):]
    return post_processing(prompt, completion)


demo = gr.Blocks(
    css=custom_css
)

with demo:
    gr.Markdown(value=description)
    with gr.Row():
        input_col , settings_col  = gr.Column(scale=6), gr.Column(scale=6), 
        with input_col:
            code = gr.Code(lines=28,label='Input', value="def sieve_eratosthenes(n):")
        with settings_col:
            with gr.Accordion("Generation Settings", open=True):
                max_new_tokens= gr.Slider(
                    minimum=8,
                    maximum=128,
                    step=1,
                    value=48,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.5,
                    step=0.1,
                    value=0.2,
                    label="Temperature",
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=1.9,
                    step=0.1,
                    value=1.0,
                    label="Repetition Penalty. 1.0 means no penalty.",
                )
                seed = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    step=1,
                    label="Random Seed"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.9,
                    label="Top P",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=4,
                    label="Top K",
                )
                use_cache = gr.Checkbox(
                    label="Use Cache",
                    value=True
                )
    
    with gr.Row():
        run = gr.Button(elem_id="orange-button", value="Generate More Code")

    # with gr.Row():
    #     # _, middle_col_row_2, _ = gr.Column(scale=1), gr.Column(scale=6), gr.Column(scale=1)
    #     # with middle_col_row_2:
    #     output = gr.HTML(label="Generated Code")

    event = run.click(code_generation, [code, max_new_tokens, temperature, seed, top_p, top_k, use_cache, repetition_penalty], code, api_name="predict")

demo.queue(max_size=40).launch()