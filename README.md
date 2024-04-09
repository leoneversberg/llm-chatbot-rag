# llm-chatbot-rag

![Screenshot](/images/example.jpg)

To use certain LLM models (such as Gemma), you need to create a .env file containing the line `ACCESS_TOKEN=<your hugging face token>`

Install dependencies with `pip install -r requirements.txt`

Run with `streamlit run src/app.py`

### Using quantization requires a GPU
To use bitsandbytes quantization, a Nvidia GPU is required.
Make sure to install the NVIDIA Toolkit first and then PyTorch. 

You can check if your GPU is available in Python with
```
import torch
print(torch.cuda.is_available())
```

If you do not have a compatible GPU, try setting `device="cpu"` for the model and remove the quantization config.
