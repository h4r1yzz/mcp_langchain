import streamlit as st
import asyncio
from combine2 import get_story_and_search_results
from llm_chat import chat_with_llm

st.title("Story Writer and Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "story_generated" not in st.session_state:
    st.session_state.story_generated = False
if "current_story" not in st.session_state:
    st.session_state.current_story = ""
if "current_search" not in st.session_state:
    st.session_state.current_search = ""

if not st.session_state.story_generated:
    st.subheader("Input to Generate a Story")
    story_prompt = st.text_input("Enter a topic for your story:")
    
    if st.button("Generate Story"):
        with st.spinner("Generating story and searching..."):
            search_results, story = asyncio.run(get_story_and_search_results(story_prompt))
            
            formatted_search_results = ""
            if search_results:
                links = search_results.split('\n')
                formatted_search_results = "\n\n".join(link.strip() for link in links if link.strip())
            
            st.session_state.current_story = story
            st.session_state.current_search = formatted_search_results
            st.session_state.story_generated = True
            
            initial_content = f"""Here's your generated story and related information:

Story:
{story}

Related Links: 
\n
{formatted_search_results}
"""
            
            st.session_state.messages.append({"role": "assistant", "content": initial_content})
            st.rerun()

if st.session_state.story_generated:
    st.subheader("Chat about the Story")
    
    with st.expander("View Story and Related Links", expanded=False):
        st.markdown("**Generated Story:**")
        st.markdown(st.session_state.current_story)
        st.markdown("---")
        st.markdown("**Related Links:**")
        if st.session_state.current_search:
            for link in st.session_state.current_search.split('\n\n'):
                if link.strip():
                    st.markdown(f"{link.strip()}")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about the story..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            contextualized_prompt = f"""Based on this story:
{st.session_state.current_story}

And these related links:
{st.session_state.current_search}

User question: {prompt}"""
            
            response = asyncio.run(chat_with_llm(contextualized_prompt))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Generate New Story"):
        st.session_state.story_generated = False
        st.session_state.current_story = ""
        st.session_state.current_search = ""
        st.session_state.messages = []
        st.rerun() 