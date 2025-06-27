# huggingface_chat
Chat with any model available in Huggingface's transformers library inside of a gradio interface locally.

You will need to create an account on [Huggingface](https://huggingface.co) and create a user access token for your account [here](https://huggingface.co/settings/tokens). Place this token in the .env file, see the example provided.

Use uv to create a virtual environment and install all packages.

If you do not have uv installed, you can install it by following this [link](https://docs.astral.sh/uv/getting-started/installation/).

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

To run the script:

```
uv run chat.py
```

Navigate a browser to localhost:7860 to chat with the model you have chosen.

Model chosen is specified as the MODEL_VARIANT variable near the top of the chat.py file. Users can choose 4 or 8 bit quantization to run a larger model on a smaller card.


