import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import random
import mimetypes

# Add your Google API key here
GOOGLE_API_KEY = "AIzaSyCdCvl0nOS539oNE27kPN1xwtoBAHhOQes"

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Function to convert the messages format from Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "file":
                gemini_message["parts"].append(content["file_content"])

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

# Function to query and stream the response from the LLM
def stream_llm_response(model_params):
    response_message = ""

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name = model_params["model"],
        generation_config={
            "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
        }
    )
    gemini_messages = messages_to_gemini(st.session_state.messages)

    for chunk in model.generate_content(
        contents=gemini_messages,
        stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="DadI Custom Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The DadI custom chatbot</i> 💬</h1>""")


    st.markdown("""
    This is Open Source. We do not share any private information.
    For recommendations, email me at dadynasser89@gmail.com or text me at 0689924886.
    Have a good chat!
    """)
    
    # --- Main Content ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the previous messages if there are any
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":      
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "file":
                    st.text(f"File: {content['file_name']} ({content['file_type']})")
                    with st.expander("View file content"):
                        st.code(content['file_content'], language=content['file_type'].split('/')[-1])

    # Side bar model options and inputs
    with st.sidebar:
        st.divider()
        
        model = st.selectbox("Select a model:", google_models, index=0)
        
        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        def reset_conversation():
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                st.session_state.pop("messages", None)

        st.button(
            "🗑️ Reset conversation", 
            on_click=reset_conversation,
        )

        st.divider()

        # Image Upload
        st.write("### **🖼️ Add an image or a video file:**")

        def add_image_to_messages():
            if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                if img_type == "video/mp4":
                    # save the video file
                    video_id = random.randint(100000, 999999)
                    with open(f"video_{video_id}.mp4", "wb") as f:
                        f.write(st.session_state.uploaded_img.read())
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "video_file",
                                "video_file": f"video_{video_id}.mp4",
                            }]
                        }
                    )
                else:
                    raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                    img = get_image_base64(raw_img)
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )

        cols_img = st.columns(2)

        with cols_img[0]:
            with st.popover("📁 Upload"):
                st.file_uploader(
                    "Upload an image or a video:", 
                    type=["png", "jpg", "jpeg", "mp4"], 
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:                    
            with st.popover("📸 Camera"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    st.camera_input(
                        "Take a picture", 
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        # File Upload
        st.write("### **📄 Add a file:**")
        
        def add_file_to_messages():
            if st.session_state.uploaded_file:
                file = st.session_state.uploaded_file
                file_content = file.getvalue().decode("utf-8")
                mime_type, _ = mimetypes.guess_type(file.name)
                
                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "file",
                            "file_name": file.name,
                            "file_type": mime_type or "text/plain",
                            "file_content": file_content
                        }]
                    }
                )

        st.file_uploader(
            "Upload a file:", 
            type=None,  # Allow all file types
            accept_multiple_files=False,
            key="uploaded_file",
            on_change=add_file_to_messages,
        )

    # Chat input
    if prompt := st.chat_input("Hi! Ask me anything..."):
        st.session_state.messages.append(
            {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": prompt,
                }]
            }
        )
        
        # Display the new messages
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(
                stream_llm_response(model_params=model_params)
            )

if __name__=="__main__":
    main()