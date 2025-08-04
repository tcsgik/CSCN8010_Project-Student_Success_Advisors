import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import streamlit as st
from datetime import datetime

from src.chatbotController import ChatbotController

# Page config
st.set_page_config(page_title="Conestoga Student Support Chatbot", page_icon="💬")

# Hide Streamlit footer
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# Custom CSS to make layout responsive
# Custom CSS
st.markdown("""
    <style>
        .chat-input {
            border-top: 1px solid #eee !important;
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Student Support Chatbot")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotController()

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = [{
                "sender": "Lulu",
                "role": "Student Success Advisor",
                "avatar": "💁‍♀️",
                "text": "Hello, my name is Lulu, how can I help you today?",
                "time": datetime.now().strftime("%H:%M")
            }]

# If input was previously stored, clear it before rendering the input box
if "clear_input" in st.session_state:
    st.session_state.chat_input = ""
    del st.session_state["clear_input"]

# --- Build the chat history HTML ---
chat_body = ""

for msg in st.session_state.history:
    if msg.get("sender") == "System":
        chat_body += f"<div style='text-align:center; color: gray; font-size: 12px; margin: 10px 0;'>{msg['text']}</div>"
    else:
        align = "left" if msg["sender"] != "You" else "right"
        bubble_color = "#daf4fa"
        avatar = msg.get("avatar", "👤")
        chat_body += f"<div style='display:flex; flex-direction:{'row' if align == 'left' else 'row-reverse'}; margin-bottom:10px;'><div style='font-size:24px; margin:0 10px;'>{avatar}</div><div><div style='font-weight:bold; font-size:13px;color:lightgray;'>{msg['sender']}</div><div style='font-size:11px; color:gray;'>{msg.get('role', '')}</div><div style='background-color:{bubble_color}; color:black; padding:10px; border-radius:10px; max-width:600px; margin-top:4px;'>{msg['text']}</div></div></div>"

# Scroll to bottom using JavaScript
full_html = f"<div id='chat-box' style='height: 400px; overflow-y: auto; border: 1px solid #ccc; padding-right: 10px; border-radius: 10px;'>{chat_body}</div><script>var chatBox = document.getElementById('chat-box'); if (chatBox) {{ chatBox.scrollTop = chatBox.scrollHeight; }}</script>"

# Render using components.html (not st.markdown)
components.html(full_html, height=400)

# Show "Advisor is typing..." message if a response is pending
if "pending_input" in st.session_state:
    st.markdown(
        "<div style='color: gray; font-style: italic; padding: 5px 0;'>💁‍♀️ Lulu is typing ...</div>",
        unsafe_allow_html=True
    )

# --- Chat input ---
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
user_input = st.text_input("Type a message here and press Enter...", label_visibility="collapsed", key="chat_input")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")  # Remove to avoid reprocessing
    reply = st.session_state.chatbot.get_answer(user_input)

    st.session_state.history.append({
        "sender": "Lulu",
        "role": "Student Success Advisor",
        "avatar": "💁‍♀️",
        "text": reply,
        "time": datetime.now().strftime("%H:%M")
    })
    st.rerun()

if user_input:
    st.session_state.history.append({
        "sender": "You",
        "avatar": "🧑‍🎓",
        "text": user_input,
        "time": datetime.now().strftime("%H:%M")
    })

    # Store input to handle bot reply on next run
    st.session_state["pending_input"] = user_input
    st.session_state["clear_input"] = True  
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # Close chat-input
