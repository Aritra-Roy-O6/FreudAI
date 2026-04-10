import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Fourth Order | Track B", page_icon="🧠", layout="centered")

st.title("FreudAI: Your AI-Powered Emotional Support Companion")
st.subheader("AI-Powered Mental Health & Emotional Support")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally show the debug tag for the AI's responses
        if message["role"] == "assistant" and "tag" in message:
            st.caption(f"*Routed as: {message['tag']}*")

# Handle user input
if prompt := st.chat_input("What's on your mind?"):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            response = requests.post(API_URL, json={"message": prompt})
            response.raise_for_status()
            data = response.json()
            
            bot_reply = data.get("response", "Error generating response.")
            emotion_tag = data.get("emotion_tag", "[UNKNOWN]")
            
            message_placeholder.markdown(bot_reply)
            st.caption(f"*Routed as: {emotion_tag}*")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_reply,
                "tag": emotion_tag
            })
            
        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection failed: {e}. Is your FastAPI server running?")