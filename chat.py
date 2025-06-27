import os
import json
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline
import torch

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')


MODEL_VARIANT = "google/txgemma-9b-chat" 

def create_chat_interface(pipeline_model, tokenizer):

    def predict(message,history):
        messages = history + [{"role": "user", "content": message}]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = pipe(prompt, max_new_tokens=512, return_full_text=False)[0]['generated_text']
        return output

    interface = gr.ChatInterface(fn=predict,type='messages',
                                )
    return interface
        


if __name__ == '__main__':
    model_id = f"{MODEL_VARIANT}"
    
    if MODEL_VARIANT == "google/txgemma-2b-predict":
        additional_args = {}
    else:
        additional_args = {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True
                # load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_compute_dtype=torch.bfloat16,
                # bnb_4bit_quant_type="nf4"
            )
        }
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        **additional_args,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device="cuda",
    )
    chat_app = create_chat_interface(pipe,tokenizer)
    chat_app.launch(server_port=7860)
