import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# Load the models and tokenizers
translation_model_name_de = "Helsinki-NLP/opus-mt-en-de"
translation_model_name_fr = "Helsinki-NLP/opus-mt-en-fr"
translation_model_name_hi = "Helsinki-NLP/opus-mt-en-hi"
tokenizer_de = MarianTokenizer.from_pretrained(translation_model_name_de)
tokenizer_fr = MarianTokenizer.from_pretrained(translation_model_name_fr)
tokenizer_hi = MarianTokenizer.from_pretrained(translation_model_name_hi)

translation_model_de = MarianMTModel.from_pretrained(translation_model_name_de)
translation_model_fr = MarianMTModel.from_pretrained(translation_model_name_fr)
translation_model_hi = MarianMTModel.from_pretrained(translation_model_name_hi)

def translate_to_german(data):
    inputs = tokenizer_de(data, return_tensors="pt", padding=True, truncation=True)
    translated = translation_model_de.generate(**inputs)
    translated_text = tokenizer_de.decode(translated[0], skip_special_tokens=True)
    return translated_text

def translate_to_french(data):
    inputs = tokenizer_fr(data, return_tensors="pt", padding=True, truncation=True)
    translated = translation_model_fr.generate(**inputs)
    translated_text = tokenizer_fr.decode(translated[0], skip_special_tokens=True)
    return translated_text

def translate_to_hindi(data):
    inputs = tokenizer_hi(data, return_tensors="pt", padding=True, truncation=True)
    translated = translation_model_hi.generate(**inputs)
    translated_text = tokenizer_hi.decode(translated[0], skip_special_tokens=True)
    return translated_text



def translate_text(text, target_language):
    if target_language == "German":
        processed_text = translate_to_german(text)
    elif target_language == "French":
        processed_text = translate_to_french(text)
    elif target_language == "Hindi":
        processed_text = translate_to_hindi(text)
    return processed_text

# Define input and output components
textbox = gr.Textbox(lines=5, label="Input Text")
radio = gr.Radio(["German", "French", "Hindi"], label="Target Language")
output_text = gr.Textbox(label="Translated Text")

iface = gr.Interface(fn=translate_text, inputs=[textbox, radio], outputs=output_text, title="Translation")
iface.launch(share=False)
