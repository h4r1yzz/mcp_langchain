import streamlit as st
import asyncio
from combine2 import get_search_and_chat_results
import io
import pandas as pd
import welly
import numpy as np
import lasio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tempfile
import os

st.title("Chat Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "well_data" not in st.session_state:
    st.session_state.well_data = {}

# template for creating well log plots for the llm
def create_well_log_plot(well_data, curves_to_plot, title="Well Log Plot"):
    try:
        depth = np.array(list(well_data['curve_data'].values())[0])
        
        # Create figure and grid
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, len(curves_to_plot), figure=fig)
        
        # Plot each curve
        for i, curve_name in enumerate(curves_to_plot):
            if curve_name not in well_data['curve_data']:
                continue
                
            ax = fig.add_subplot(gs[0, i])
            curve_data = well_data['curve_data'][curve_name]
            unit = well_data['units'].get(curve_name, '')
            
            ax.plot(curve_data, depth)
            ax.set_ylim(max(depth), min(depth))  
            ax.grid(True)
            ax.set_xlabel(f"{curve_name} ({unit})")
            
            if i == 0:
                ax.set_ylabel("Depth")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def process_las_file(uploaded_file):
    try:
        # First try using lasio for basic LAS file parsing
        content = uploaded_file.read()
        uploaded_file.seek(0)  
        
        # Parse with lasio first
        las = lasio.read(io.StringIO(content.decode('utf-8')))
        
        # Extract basic information using lasio
        well_info = {
            "name": las.well.WELL.value if hasattr(las.well, 'WELL') else uploaded_file.name,
            "header": {
                item.mnemonic: str(item.value) for item in las.well
            },
            "curves": [curve.mnemonic for curve in las.curves],
            "curve_data": {curve.mnemonic: curve.data.tolist() for curve in las.curves},
            "units": {curve.mnemonic: curve.unit for curve in las.curves},
            "depth_range": f"{min(las.index):.2f} - {max(las.index):.2f}",
            "step": f"{las.index[1] - las.index[0]:.2f}" if len(las.index) > 1 else "N/A"
        }
        
        try:
            # Create a temporary file to save the LAS content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.las', delete=False) as tmp_file:
                tmp_file.write(content.decode('utf-8'))
                tmp_file.flush()
                
                # Now load the temporary file with welly
                well = welly.Well.from_las(tmp_file.name)
                
                if hasattr(well, 'location'):
                    well_info['location'] = well.location
                if hasattr(well, 'uwi'):
                    well_info['uwi'] = well.uwi
                    
                os.unlink(tmp_file.name)
                
        except Exception as welly_error:
            st.warning(f"Note: Some advanced well analysis features may be limited. Welly error: {str(welly_error)}")
        
        # Create a summary of the well data
        summary = f"""Well Name: {well_info['name']}
Depth Range: {well_info['depth_range']} {las.curves[0].unit if las.curves else 'unknown'}
Depth Step: {well_info['step']} {las.curves[0].unit if las.curves else 'unknown'}
Available Curves: {', '.join(well_info['curves'])}
"""

        if well_info['header']:
            summary += "\nWell Header Information:\n"
            for key, value in well_info['header'].items():
                if value and str(value).strip():
                    summary += f"{key}: {value}\n"
        
        return well_info, summary

    except Exception as e:
        st.error(f"Error processing LAS file {uploaded_file.name}: {str(e)}")
        if "curves" in str(e):
            st.info("Tip: The file might be missing curve data or might have an unexpected format. Please check the file structure.")
        elif "encoding" in str(e):
            st.info("Tip: The file might have an unexpected encoding. Try converting it to UTF-8.")
        return None, f"Error processing LAS file {uploaded_file.name}: {str(e)}"

def process_uploaded_file(uploaded_file):
    # Process uploaded file and return its content as string
    if uploaded_file is None:
        return None
    
    file_type = uploaded_file.type
    try:
        if file_type.startswith('text/'):
            content = uploaded_file.getvalue().decode('utf-8')
            return content
        elif file_type == 'application/pdf':
            return f"PDF file uploaded: {uploaded_file.name}"
        elif 'spreadsheet' in file_type or file_type == 'application/vnd.ms-excel':
            df = pd.read_excel(uploaded_file) if 'excel' in file_type else pd.read_csv(uploaded_file)
            return df.to_string()
        elif uploaded_file.name.lower().endswith('.las'):
            # Process LAS file and store well data
            well_data, summary = process_las_file(uploaded_file)
            if well_data:
                st.session_state.well_data[uploaded_file.name] = well_data
            return summary
        else:
            return f"File uploaded: {uploaded_file.name} (type: {file_type})"
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return f"Error processing file {uploaded_file.name}: {str(e)}"

# File Upload Section
with st.sidebar:
    st.header("File Upload")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["txt", "pdf", "csv", "xlsx", "xls", "las"],
        accept_multiple_files=True,
        help="Upload files to discuss with the chatbot. Supports well log (.las) files!"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_key = f"{uploaded_file.name}_{uploaded_file.type}"
            if file_key not in st.session_state.uploaded_files:
                content = process_uploaded_file(uploaded_file)
                st.session_state.uploaded_files[file_key] = {
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "content": content
                }
        
        st.subheader("Uploaded Files")
        for file_key, file_info in st.session_state.uploaded_files.items():
            st.write(f"📄 {file_info['name']}")

# Chat Section
st.subheader("Chat here")

# # Display uploaded files content in a collapsible section
# if st.session_state.uploaded_files:
#     with st.expander("View Uploaded Files", expanded=False):
#         for file_key, file_info in st.session_state.uploaded_files.items():
#             st.markdown(f"**{file_info['name']}**")
            

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about the well log data..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        context = "Available Well Log Data:\n\n"
        for file_name, well_data in st.session_state.well_data.items():
            context += f"Well: {file_name}\n"
            context += f"Depth Range: {well_data['depth_range']}\n"
            context += f"Available Curves: {', '.join(well_data['curves'])}\n"
            context += f"Units: {', '.join(f'{k}: {v}' for k, v in well_data['units'].items())}\n\n"
        
        # Add file contents
        if st.session_state.uploaded_files:
            context += "\nUploaded Files Content:\n"
            for file_info in st.session_state.uploaded_files.values():
                context += f"\n{file_info['name']}:\n{file_info['content']}\n"

        # Get response from LLM
        search_results, response = asyncio.run(get_search_and_chat_results(prompt, context))
        
        # Check if the response contains a plotting request
        if any(keyword in prompt.lower() for keyword in ['plot', 'graph', 'visualize', 'display', 'show']):
            try:
                # Extract curve names from the response or prompt
                curve_names = []
                for well_name, well_data in st.session_state.well_data.items():
                    for curve in well_data['curves']:
                        if curve.lower() in prompt.lower() or curve.lower() in response.lower():
                            curve_names.append(curve)
                
                if curve_names:
                    # Create plot for each well that has the requested curves
                    for well_name, well_data in st.session_state.well_data.items():
                        valid_curves = [curve for curve in curve_names if curve in well_data['curves']]
                        if valid_curves:
                            fig = create_well_log_plot(well_data, valid_curves, f"Well Log Plot - {well_name}")
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)  
                                st.markdown(f"I've created a plot showing the following curves for {well_name}: {', '.join(valid_curves)}")
                else:
                    st.warning("I couldn't identify which curves to plot. Please specify the curve names you'd like to see.")
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
        
        # Format the response with search results if available
        formatted_response = response if response else "I apologize, but I couldn't process your request. Please try again."
        if search_results and search_results.strip():
            links = search_results.split('\n')
            formatted_links = [f"{link.strip()}" for link in links if link.strip()]
            if formatted_links:
                formatted_search_results = "\n".join(formatted_links)
                formatted_response = f"{formatted_response}\n\nRelevant Links:\n{formatted_search_results}"
        
        st.markdown(formatted_response)
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

# # Reset button
# if st.button("Clear Uploaded Files"):
#     st.session_state.uploaded_files = {}
#     st.session_state.messages = []
#     st.session_state.well_data = {}
#     st.rerun()