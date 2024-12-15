import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="Simple Streamlit App", page_icon=":smiley:", layout="centered")

# Title of the app
st.title("Welcome to My Simple Streamlit App!")

# Text input for user name
name = st.text_input("Enter your name:")

# Display greeting message
if name:
    st.write(f"Hello, {name}! Nice to meet you!")
