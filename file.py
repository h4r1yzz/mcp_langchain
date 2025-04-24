from mcp.server.fastmcp import FastMCP
import os
import sys
import io
import contextlib
from typing import Dict, List, Any, Union, Optional, Tuple
import lasio
import numpy as np
import pandas as pd
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For potential use in crossplots
from plotly.utils import PlotlyJSONEncoder

# Global variable to store file paths
LAS_FILE_PATHS: List[str] = []

# Create an MCP server
mcp = FastMCP("LAS File Analyzer")

@mcp.tool()
def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Analyze a LAS file and return comprehensive metadata and curve data.
    
    Args:
        file_path: Path to the LAS file
        
    Returns:
        Dictionary containing metadata and curve data from the LAS file
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    # Add file path to global list if not already present
    if file_path not in LAS_FILE_PATHS:
        LAS_FILE_PATHS.append(file_path)
    
    try:
        # Read the LAS file
        las = lasio.read(file_path)
        
        # Initialize metadata structure
        metadata = {
            "file_info": {
                "file_path": file_path,
                "file_size_bytes": os.path.getsize(file_path),
                "tracked_files": LAS_FILE_PATHS
            },
            "version_info": {},
            "well_info": {},
            "curves_summary": {
                "count": len(las.curves),
                "curves": []
            },
            "parameters": {
                "count": len(las.params) if hasattr(las, "params") else 0,
                "params": []
            },
            "data_summary": {
                "rows": len(las.data) if hasattr(las, "data") and las.data is not None else 0,
                "columns": len(las.curves) if hasattr(las, "curves") else 0
            },
            "curve_data": {}
        }
        
        # # Extract version info
        # if hasattr(las, "version"):
        #     version_dict = {}
        #     if hasattr(las.version, "value"): version_dict["version"] = las.version.value
        #     if hasattr(las.version, "WRAP"): version_dict["wrap"] = las.version.WRAP
        #     if hasattr(las.version, "DLM"): version_dict["dlm"] = las.version.DLM
        #     metadata["version_info"] = version_dict
        
        # Extract well info with descriptions
        if hasattr(las, "well"):
            well_dict = {}
            for k, v in las.well.items():
                try:
                    well_dict[k] = {
                        "value": str(v.value) if hasattr(v, "value") else str(v),
                        "description": v.descr if hasattr(v, "descr") else ""
                    }
                except:
                    well_dict[k] = {"value": "Error extracting value", "description": ""}
            metadata["well_info"] = well_dict
        
        # Extract API codes and parameter info from the file
        api_codes, param_info = {}, {}
        
        # Extract API codes from the file
        try:
            with open(file_path, 'r') as f:
                # Process curve section for API codes
                line = f.readline()
                while line and not line.strip().startswith('~C'): line = f.readline()
                if line: line = f.readline()  # Skip ~C line
                
                # Read curve definitions
                while line and not line.strip().startswith('~'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) >= 1:
                            curve_def = parts[0].strip()
                            mnem_parts = curve_def.split('.')
                            if len(mnem_parts) >= 2:
                                mnem = mnem_parts[0].strip()
                                rest = mnem_parts[1].strip()
                                unit_api_parts = rest.split()
                                api_code = " ".join(unit_api_parts[1:]) if len(unit_api_parts) > 1 else ""
                                api_codes[mnem] = api_code
                    line = f.readline()
                
                # Process parameter section
                f.seek(0)
                line = f.readline()
                while line and not line.strip().startswith('~P'): line = f.readline()
                if line: line = f.readline()  # Skip ~P line
                
                # Read parameter definitions
                while line and not line.strip().startswith('~'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) >= 1:
                            param_def = parts[0].strip()
                            description = parts[1].strip() if len(parts) > 1 else ""
                            parts = param_def.split('.')
                            if len(parts) >= 2:
                                mnem = parts[0].strip()
                                rest = parts[1].strip()
                                unit = rest.split(' ', 1)[0].strip() if ' ' in rest else ""
                                param_info[mnem] = {"unit": unit, "description": description}
                    line = f.readline()
        except Exception as e:
            print(f"Warning: Could not extract file section data: {str(e)}")
        
        # Process curves
        curve_names = []
        for curve in las.curves:
            try:
                mnemonic = curve.mnemonic if hasattr(curve, "mnemonic") else f"curve_{len(curve_names)}"
                curve_names.append(mnemonic)
                
                metadata["curves_summary"]["curves"].append({
                    "name": mnemonic,
                    "description": curve.descr if hasattr(curve, "descr") else "",
                    "unit": curve.unit if hasattr(curve, "unit") else "",
                    "api_code": api_codes.get(mnemonic, "")
                })
            except:
                continue
        
        # Process parameters
        if hasattr(las, "params"):
            for param in las.params:
                try:
                    mnemonic = param.mnemonic if hasattr(param, "mnemonic") else ""
                    if not mnemonic: continue
                    
                    metadata["parameters"]["params"].append({
                        "name": mnemonic,
                        "value": str(param.value) if hasattr(param, "value") else "",
                        "unit": param_info.get(mnemonic, {}).get("unit", ""),
                        "description": param_info.get(mnemonic, {}).get("description", "")
                    })
                except:
                    continue
        
        # Process curve data
        try:
            df = las.df()
            
            # Process each curve
            for curve_info in metadata["curves_summary"]["curves"]:
                curve_name = curve_info["name"]
                try:
                        # Special handling for DEPT curve
                        if curve_name == "DEPT":
                            # Read DEPT values directly from file
                            all_dept_values = []
                            with open(file_path, 'r') as f:
                                line = f.readline()
                                while line and not line.strip().startswith('~A'): line = f.readline()
                                if line: line = f.readline()  # Skip ~A line
                                
                                while line:
                                    parts = line.strip().split()
                                    if parts and len(parts) > 0:
                                        try: all_dept_values.append(float(parts[0]))
                                        except: pass
                                    line = f.readline()
                            
                            if all_dept_values:
                                min_val = min(all_dept_values)
                                max_val = max(all_dept_values)
                                mean_val = sum(all_dept_values) / len(all_dept_values)
                                valid_points = len(all_dept_values)
                                
                                # # Get first 15 points
                                # first_points = []
                                # for i in range(min(15, len(all_dept_values))):
                                #     first_points.append(all_dept_values[i])
                                
                                # # Get sparse samples (every 500 points)
                                # sparse_points = []
                                # if len(all_dept_values) > 500:
                                #     for i in range(0, len(all_dept_values), 500):
                                #         if i >= 15 and i < len(all_dept_values) - 5:  # Skip if already in first or last points
                                #             sparse_points.append(all_dept_values[i])
                                
                                # # Get last 5 points
                                # last_points = []
                                # for i in range(max(0, len(all_dept_values) - 5), len(all_dept_values)):
                                #     last_points.append(all_dept_values[i])
                                
                                metadata["curve_data"][curve_name] = {
                                    "min": min_val,
                                    "max": max_val,
                                    "mean": mean_val,
                                    "valid_points": valid_points
                                }
                            else:
                                metadata["curve_data"][curve_name] = {
                                    "min": None,
                                    "max": None,
                                    "mean": None,
                                    "valid_points": 0
                                }
                        # Handle other curves
                        elif curve_name in df.columns:
                            curve_data = df[curve_name].values
                            depth_data = df.index.values if hasattr(df, 'index') else las.curves[0].data
                            
                            # Filter out NaN values
                            valid_indices = []
                            valid_data = []
                            for i, val in enumerate(curve_data):
                                if not (np.isnan(val) if hasattr(np, "isnan") else (val != val)):
                                    valid_indices.append(i)
                                    valid_data.append(val)
                            
                            # Calculate statistics
                            min_val = max_val = mean_val = None
                            if valid_data:
                                try:
                                    min_val = float(min(valid_data))
                                    max_val = float(max(valid_data))
                                    mean_val = float(sum(valid_data) / len(valid_data))
                                except: pass
                            
                            # # Get first 15 valid points with depths
                            # first_points = []
                            # for i in range(min(15, len(valid_indices))):
                            #     idx = valid_indices[i]
                            #     first_points.append({
                            #         "depth": float(depth_data[idx]) if idx < len(depth_data) else None,
                            #         "value": float(curve_data[idx])
                            #     })
                            
                            # # Get sparse samples (every 500 valid points)
                            # sparse_points = []
                            # if len(valid_indices) > 500:
                            #     for i in range(0, len(valid_indices), 500):
                            #         if i >= 15 and i < len(valid_indices) - 5:  # Skip if already in first or last points
                            #             idx = valid_indices[i]
                            #             sparse_points.append({
                            #                 "depth": float(depth_data[idx]) if idx < len(depth_data) else None,
                            #                 "value": float(curve_data[idx])
                            #             })
                            
                            # # Get last 5 valid points with depths
                            # last_points = []
                            # for i in range(max(0, len(valid_indices) - 5), len(valid_indices)):
                            #     idx = valid_indices[i]
                            #     last_points.append({
                            #         "depth": float(depth_data[idx]) if idx < len(depth_data) else None,
                            #         "value": float(curve_data[idx])
                            #     })
                            
                            metadata["curve_data"][curve_name] = {
                                "min": min_val,
                                "max": max_val,
                                "mean": mean_val,
                                "valid_points": len(valid_data)
                            }
                except Exception as e:
                    metadata["curve_data"][curve_name] = {"error": str(e)}
        except Exception as e:
            # If DataFrame conversion fails, try direct curve access
            for curve_info in metadata["curves_summary"]["curves"]:
                curve_name = curve_info["name"]
                try:
                    curve = las.curves[curve_name]
                    if hasattr(curve, "data"):
                        curve_data = curve.data
                        valid_data = [x for x in curve_data if not (np.isnan(x) if hasattr(np, "isnan") else (x != x))]
                        
                        # Try to get depth data
                        depth_data = None
                        try:
                            depth_curve = las.curves[0]  # Usually the first curve is depth
                            if hasattr(depth_curve, "data"):
                                depth_data = depth_curve.data
                        except:
                            depth_data = [i for i in range(len(curve_data))]  # Fallback to indices
                        
                        # Filter out NaN values
                        valid_indices = []
                        valid_data = []
                        for i, val in enumerate(curve_data):
                            if not (np.isnan(val) if hasattr(np, "isnan") else (val != val)):
                                valid_indices.append(i)
                                valid_data.append(val)
                        
                        # Calculate statistics
                        min_val = max_val = mean_val = None
                        if valid_data:
                            try:
                                min_val = float(min(valid_data))
                                max_val = float(max(valid_data))
                                mean_val = float(sum(valid_data) / len(valid_data))
                            except: pass
                        
                        # # Get first 15 valid points with depths
                        # first_points = []
                        # for i in range(min(15, len(valid_indices))):
                        #     idx = valid_indices[i]
                        #     first_points.append({
                        #         "depth": float(depth_data[idx]) if depth_data and idx < len(depth_data) else None,
                        #         "value": float(curve_data[idx])
                        #     })
                        
                        # # Get sparse samples (every 500 valid points)
                        # sparse_points = []
                        # if len(valid_indices) > 500:
                        #     for i in range(0, len(valid_indices), 500):
                        #         if i >= 15 and i < len(valid_indices) - 5:  # Skip if already in first or last points
                        #             idx = valid_indices[i]
                        #             sparse_points.append({
                        #                 "depth": float(depth_data[idx]) if depth_data and idx < len(depth_data) else None,
                        #                 "value": float(curve_data[idx])
                        #             })
                        
                        # # Get last 5 valid points with depths
                        # last_points = []
                        # for i in range(max(0, len(valid_indices) - 5), len(valid_indices)):
                        #     idx = valid_indices[i]
                        #     last_points.append({
                        #         "depth": float(depth_data[idx]) if depth_data and idx < len(depth_data) else None,
                        #         "value": float(curve_data[idx])
                        #     })
                        
                        metadata["curve_data"][curve_name] = {
                            "min": min_val,
                            "max": max_val,
                            "mean": mean_val,
                            "valid_points": len(valid_data)
                        }
                except Exception as curve_e:
                    metadata["curve_data"][curve_name] = {"error": f"Failed to extract curve data: {str(curve_e)}"}
            
            # Add error information about DataFrame conversion
            metadata["data_access_error"] = str(e)
        
        return metadata
    except Exception as e:
        return {"error": f"Error analyzing LAS file: {str(e)}"}

@mcp.tool()
def get_visualization(
    file_path: str,
    visualization_type: str,  # "multi_track", "crossplot", "heatmap"
    curve_names: List[str],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate visualizations of well log data using Plotly.

    Args:
        file_path: Path to the LAS file.
        visualization_type: Type of visualization to generate.
            Options:
              - "multi_track": Track-based log plot with multiple curves.
              - "crossplot": Scatter plot comparing two curves.
              - "heatmap": Heatmap visualization of multiple curves.
        curve_names: List of curve names to visualize.
        options: Optional dictionary of visualization options. Supported keys:
            - depth_range: [min_depth, max_depth] to limit the plot.
            - colors: List of colors for curves (for multi_track).
            - title: Title for the plot.
            - x_curve, y_curve: For crossplot (first two curves used by default).
            - color_curve: For crossplot color coding.
            - colormap: For heatmap (default: "viridis").
            - track_widths: List of relative widths for multi_track.

    Returns:
        Dictionary containing:
          - plot_id: Unique identifier for the plot.
          - plot_json_path: Path to the JSON file containing the Plotly data.
          - visualization_type: The type of visualization generated.
          - curves: List of curves included in the visualization.
          - metadata: Additional information (such as depth range and curve units).
    """
    # Get default options
    if options is None:
        options = {}

    # Use the existing LAS file analyzer to get metadata and curve information
    las_data = las_file_analyzer(file_path)
    if "error" in las_data:
        return {"error": las_data["error"]}

    # Generate a unique ID for the visualization
    timestamp = int(time.time())
    plot_id = f"{visualization_type}_{timestamp}"

    # Validate requested curves based on metadata
    available_curves = {curve["name"]: curve for curve in las_data["curves_summary"]["curves"]}
    for curve in curve_names:
        if curve not in available_curves:
            return {"error": f"Curve '{curve}' not found in the LAS file. Available curves: {', '.join(available_curves.keys())}"}

    # Read the LAS file and convert to DataFrame for visualization
    try:
        las = lasio.read(file_path)
        df = las.df()
    except Exception as e:
        return {"error": f"Error reading LAS file for visualization: {str(e)}"}
    
    # Apply depth range filter if specified
    depth_range = options.get("depth_range", None)
    if depth_range and len(depth_range) == 2:
        min_depth, max_depth = depth_range
        df = df[(df.index >= min_depth) & (df.index <= max_depth)]

    # Get title for the plot
    title = options.get("title", f"Well Log Visualization - {visualization_type.title()}")

    fig = None  # Initialize figure

    if visualization_type.lower() == "multi_track":
        n_tracks = len(curve_names)
        colors = options.get("colors", [f"rgba({i*40 % 255}, {i*70 % 255}, {i*90 % 255}, 1)" for i in range(n_tracks)])

        # Create subplots: 1 row, n_tracks columns; share y-axis (depth)
        fig = make_subplots(rows=1, cols=n_tracks, shared_yaxes=True,
                            horizontal_spacing=0.02,
                            subplot_titles=[f"{curve} ({available_curves[curve].get('unit', '')})" for curve in curve_names])

        # Plot each curve in its own track
        for i, curve in enumerate(curve_names):
            # For multi_track, skip plotting the "DEPT" curve if it's not a column in df,
            # because depth is used as the common y-axis.
            if curve == "DEPT" and curve not in df.columns:
                continue

            # Curve existence was already validated above
            curve_data = df[curve].values

            depth_data = df.index.values

            # Add trace for the curve
            fig.add_trace(
                go.Scatter(
                    x=curve_data,
                    y=depth_data,
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    name=curve
                ),
                row=1, col=i+1
            )
            # No need for x-axis titles since we have subplot titles
            fig.update_xaxes(showticklabels=True, row=1, col=i+1)

            # Add annotation for statistics if available
            if curve in las_data["curve_data"]:
                stats = las_data["curve_data"][curve]
                if stats.get("min") is not None and stats.get("max") is not None:
                    annotation_text = f"Min: {stats['min']:.2f}<br>Max: {stats['max']:.2f}"
                    # For the first subplot, use "x domain" (no number) per Plotly conventions
                    if i == 0:
                        xref_val = "x domain"
                    else:
                        xref_val = f"x{i+1} domain"
                    fig.add_annotation(dict(
                        xref=xref_val,
                        yref="paper",
                        x=0.5,
                        y=0.02,
                        text=annotation_text,
                        showarrow=False,
                        bgcolor="white",
                        opacity=0.7
                    ), row=1, col=i+1)

        # Invert y-axis (depth increasing downward)
        fig.update_yaxes(autorange="reversed")
        # Set common y-label for depth (applied to the first subplot)
        fig.update_yaxes(title_text="Depth", row=1, col=1)
        # Set overall title and layout with compact dimensions
        # Reduce margins and set appropriate height and width
        # Use zero left margin to remove the gap completely
        fig.update_layout(title=title, showlegend=False, height=400,
                          width=max(600, 150 * n_tracks),
                          margin=dict(l=0, r=40, t=50, b=40))

    elif visualization_type.lower() == "crossplot":
        if len(curve_names) < 2:
            return {"error": "Crossplot requires at least two curves"}
        x_curve = options.get("x_curve", curve_names[0])
        y_curve = options.get("y_curve", curve_names[1])

        # Validate curves
        if x_curve not in available_curves:
            return {"error": f"X-axis curve '{x_curve}' not found"}
        if y_curve not in available_curves:
            return {"error": f"Y-axis curve '{y_curve}' not found"}
        
        # Get color curve if specified
        color_curve = options.get("color_curve", None)
        fig = go.Figure()

        # Set up data for x, y, and optional color
        x_data = df[x_curve].values
        y_data = df[y_curve].values
        marker_dict = dict(size=7, opacity=0.7)
        
        # Create scatter plot
        if color_curve:
            if color_curve not in available_curves:
                return {"error": f"Color curve '{color_curve}' not found"}
            color_data = df[color_curve].values
            marker_dict.update(color=color_data, colorscale=options.get("colormap", "viridis"),
                               colorbar=dict(title=f"{color_curve} ({available_curves[color_curve].get('unit', '')})"))

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=marker_dict
        ))
        # Set axis labels and title
        x_unit = available_curves[x_curve].get("unit", "")
        y_unit = available_curves[y_curve].get("unit", "")
        fig.update_layout(
            title=title,
            xaxis_title=f"{x_curve} ({x_unit})" if x_unit else x_curve,
            yaxis_title=f"{y_curve} ({y_unit})" if y_unit else y_curve,
            template="simple_white",
            height=400,
            width=500,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

    elif visualization_type.lower() == "heatmap":
        if len(curve_names) < 1:
            return {"error": "Heatmap requires at least one curve"}

        # Prepare data matrix with dimensions: (number of rows in df) x (number of curves)
        data_matrix = np.zeros((len(df), len(curve_names)))
        for i, curve in enumerate(curve_names):
            col_data = df[curve].values.astype(float)
            # Normalize the curve data to 0-1 range
            col_min = np.nanmin(col_data)
            col_max = np.nanmax(col_data)
            if col_max > col_min:
                normalized = (col_data - col_min) / (col_max - col_min)
            else:
                normalized = col_data
            data_matrix[:, i] = normalized

        depths = df.index.values
        n = len(depths)
        step = max(1, n // 20)
        y_ticks = depths[::step]

        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=curve_names,
            y=depths,
            colorscale=options.get("colormap", "viridis"),
            colorbar=dict(title="Normalized Value")
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=45),
            yaxis=dict(title="Depth", autorange="reversed"),
            height=400,
            width=550,
            template="simple_white",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        fig.update_yaxes(tickmode="array", tickvals=y_ticks, ticktext=[f"{val:.1f}" for val in y_ticks])

    else:
        return {"error": f"Unsupported visualization type: {visualization_type}"}

    # Create visualization directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Save the Plotly figure as a JSON file
    fig_dict = fig.to_dict()
    plot_json_path = os.path.join(viz_dir, f"{plot_id}.json")

    with open(plot_json_path, 'w') as f:
        json.dump({
            "plot_data": fig_dict['data'],
            "plot_layout": fig_dict['layout']
        }, f, cls=PlotlyJSONEncoder)

    metadata = {
        "depth_range": [float(df.index.min()), float(df.index.max())] if not df.empty else None,
        "curve_units": {curve: available_curves[curve].get("unit", "") for curve in curve_names}
    }
    return {
        "plot_id": plot_id,
        "plot_json_path": plot_json_path,
        "visualization_type": visualization_type,
        "curves": curve_names,
        "metadata": metadata
    }

@mcp.tool()
def get_ascii_data(file_path: str, code: str, max_output_size: int = 10000) -> Dict[str, Any]:
    """
    Execute Python code on a LAS file dataframe and return the results.

    Args:
        file_path: Path to the LAS file
        code: Python code to execute on the dataframe. The dataframe is available as 'df'
        max_output_size: Maximum size of the output to return (default: 10000 characters)

    Returns:
        Dictionary containing the execution results and any output
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    # To capture the output
    stdout_capture = io.StringIO()

    result = {
        "output": "",
        "result": None,
        "error": None,
        "execution_time": 0
    }

    try:
        # Read the LAS file
        start_time = time.time()
        las = lasio.read(file_path)
        df = las.df()

        # Execute the code with captured stdout
        with contextlib.redirect_stdout(stdout_capture):
            # Create a local namespace with the dataframe
            local_namespace = {
                'df': df,
                'np': np,
                'pd': pd,
                'las': las
            }

            # Execute the code
            exec(code, {}, local_namespace)

            # Check for a return value (last expression)
            if '_' in local_namespace:
                result["result"] = local_namespace['_']

        # Get the captured stdout
        stdout_output = stdout_capture.getvalue()
        if stdout_output:
            result["output"] = stdout_output[:max_output_size]
            if len(stdout_output) > max_output_size:
                result["output"] += "\n... (output truncated)"

        result["execution_time"] = time.time() - start_time

        # If the result is a DataFrame, convert to a string representation
        if isinstance(result["result"], pd.DataFrame):
            if len(result["result"]) > 100:
                # If DataFrame is large, show only first and last rows
                df_head = result["result"].head(50).to_string()
                df_tail = result["result"].tail(50).to_string()
                result["result"] = f"DataFrame with {len(result['result'])} rows and {len(result['result'].columns)} columns:\n\nFirst 50 rows:\n{df_head}\n\nLast 50 rows:\n{df_tail}"
            else:
                result["result"] = result["result"].to_string()
        elif isinstance(result["result"], np.ndarray):
            if result["result"].size > 100:
                # If array is large, show only first and last elements
                result["result"] = f"Array with {result['result'].size} elements:\n\nFirst 50 elements:\n{result['result'][:50]}\n\nLast 50 elements:\n{result['result'][-50:]}"
            else:
                result["result"] = str(result["result"])
        elif result["result"] is not None:
            # Convert other types to string
            result["result"] = str(result["result"])[:max_output_size]

    except Exception as e:
        result["error"] = f"Error executing code: {str(e)}"

    return result

if __name__ == "__main__":
    mcp.run(transport='stdio')