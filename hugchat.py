import openai
import gradio as gr
import os
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.llms import VertexAI
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

def process(input_type, text_input=None, audio_input=None):
    openai.api_key = openai_api_key

    if input_type == 'Text':
        if text_input:
            message = text_input
    elif input_type == 'Audio':
        if audio_input:
            audio = open(audio_input, "rb")
            # Your audio processing code here
            transcript = openai.Audio.transcribe("whisper-1", audio)
            message = transcript["text"]
    
    #llm
    if message:
        llm = GooglePalm(google_api_key=google_api_key)
        llm.temperature = 0.1
        template = "Tell me reasons why I might be having these symptoms all at the same time: {topic}. How do I treat these symptoms at home? How do I treat these symptoms?"
        prompt = PromptTemplate.from_template(template)
        prompt.format(topic = message)
        chain = LLMChain(llm=llm, prompt = prompt)
        return chain.run(message)

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Radio(choices=["Text", "Audio"], label="Input Type"),
        gr.Textbox(lines=2, placeholder="Enter Text Here...", label="Text Input"),
        gr.Audio(source="microphone", type="filepath", label="Audio Input")
    ],
    outputs="text"
)

demo.launch()
