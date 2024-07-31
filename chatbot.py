import transformers
import torch
import gradio as gr

# Load the model and tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = "microsoft/phi-2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device).to(device)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Define a function to generate responses
def generate_response(user_input):
    global messages
    messages.append({"role": 'user', 'content': user_input})

    # Prepare the prompt using the messages
    input_texts = [f"{msg['role']}: {msg['content']}" for msg in messages]
    input_text = "\n".join(input_texts)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate response
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    # Decode the response
    response = outputs[0][inputs['input_ids'].shape[-1]:]
    response_msg = tokenizer.decode(response, skip_special_tokens=True)

    messages.append({"role": 'assistant', 'content': response_msg})

    # Return the updated chat history as a list of lists
    return [[msg['role'], msg['content']] for msg in messages]


# Initialize the chat history and messages
messages = [{"role": "system",
             "content": "You are a therapist chatbot who always responds in a supportive and understanding tone!"}]

# Create the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        value=[("System", "Hi,I'm Helpify, a therapist chatbot who always responds in a supportive and understanding tone")],
        elem_id="chatbot")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
    txt.submit(generate_response, txt, chatbot, queue=False)
demo.queue()
demo.launch(debug=True)
