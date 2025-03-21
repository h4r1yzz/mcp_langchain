from mcp.server.fastmcp import FastMCP
import os
import sys
from typing import Dict, List, Any, Union, Optional, Tuple
import lasio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

# Global variable to store file paths
LAS_FILE_PATHS: List[str] = []

# Create an MCP server
mcp = FastMCP("LAS File Analyzer")

@mcp.tool()
def las_file_analyzer(file_path: str) -> Dict[str, Any]:
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
def get_specific_curve_value(file_path: str, curve_name: str, index: Optional[int] = None, depth_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Get a specific value from a curve at a given index or depth.
    
    Args:
        file_path: Path to the LAS file
        curve_name: Name of the curve to query (e.g., 'CGXT', 'GRGC')
        index: Index of the value to retrieve (e.g., 7 for the 7th value)
        depth_value: Specific depth to query (alternative to index)
        
    Returns:
        Dictionary containing the requested value and context
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if curve_name is None:
        return {"error": "Curve name must be specified"}
    
    if index is None and depth_value is None:
        return {"error": "Either index or depth_value must be specified"}
    
    try:
        # Read the LAS file
        las = lasio.read(file_path)
        
        # Find the curve
        curve = None
        for c in las.curves:
            if hasattr(c, "mnemonic") and c.mnemonic == curve_name:
                curve = c
                break
        
        if curve is None:
            return {"error": f"Curve '{curve_name}' not found in the LAS file"}
        
        # Get depth curve (usually the first curve)
        depth_curve = las.curves[0]
        
        # If searching by depth
        if depth_value is not None:
            # Find the closest depth value
            closest_idx = None
            min_diff = float('inf')
            
            for i, depth in enumerate(depth_curve.data):
                diff = abs(depth - depth_value)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx is None:
                return {"error": f"Could not find depth value close to {depth_value}"}
            
            # Get the value at the closest depth
            value = curve.data[closest_idx]
            depth = depth_curve.data[closest_idx]
            
            # Get context (values before and after)
            context_values = []
            context_depths = []
            
            # Get 2 values before and 2 values after if available
            for i in range(max(0, closest_idx - 2), min(len(curve.data), closest_idx + 3)):
                if i != closest_idx:  # Skip the main value as we already have it
                    context_values.append(float(curve.data[i]))
                    context_depths.append(float(depth_curve.data[i]))
            
            return {
                "curve_name": curve_name,
                "depth": float(depth),
                "value": float(value),
                "index": closest_idx,
                "context": {
                    "depths": context_depths,
                    "values": context_values
                },
                "unit": curve.unit if hasattr(curve, "unit") else ""
            }
        
        # If searching by index
        elif index is not None:
            if index < 0 or index >= len(curve.data):
                return {"error": f"Index {index} is out of range (0-{len(curve.data)-1})"}
            
            value = curve.data[index]
            depth = depth_curve.data[index] if index < len(depth_curve.data) else None
            
            # Get context (values before and after)
            context_values = []
            context_depths = []
            context_indices = []
            
            # Get 2 values before and 2 values after if available
            for i in range(max(0, index - 2), min(len(curve.data), index + 3)):
                if i != index:  # Skip the main value as we already have it
                    context_values.append(float(curve.data[i]))
                    context_depths.append(float(depth_curve.data[i]) if i < len(depth_curve.data) else None)
                    context_indices.append(i)
            
            return {
                "curve_name": curve_name,
                "depth": float(depth) if depth is not None else None,
                "value": float(value),
                "index": index,
                "context": {
                    "indices": context_indices,
                    "depths": context_depths,
                    "values": context_values
                },
                "unit": curve.unit if hasattr(curve, "unit") else ""
            }
        
    except Exception as e:
        return {"error": f"Error retrieving curve value: {str(e)}"}

@mcp.tool()
def visualize_well_log(
    file_path: str,
    visualization_type: str,  # "multi_track", "crossplot", "heatmap"
    curve_names: List[str],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate visualizations of well log data.
    
    Args:
        file_path: Path to the LAS file
        visualization_type: Type of visualization to generate
            - "multi_track": Track-based log plot with multiple curves
            - "crossplot": Scatter plot comparing two curves
            - "heatmap": Heatmap visualization of multiple curves
        curve_names: List of curve names to visualize
        options: Optional dictionary of visualization options
            - depth_range: [min_depth, max_depth] to limit the plot
            - colors: List of colors for curves
            - title: Title for the plot
            - x_curve, y_curve: For crossplot (first two curve_names used if not specified)
            - color_curve: For crossplot coloring
            - colormap: For heatmap (default: "viridis")
            - track_widths: For multi_track (relative widths)
            
    Returns:
        Dictionary containing:
        - image_filename: Filename of the saved visualization image
        - visualization_type: Type of visualization generated
        - curves: List of curves included in the visualization
        - metadata: Additional information about the visualization
    """
    # Get default options
    if options is None:
        options = {}
    
    # Use las_file_analyzer to get the data
    las_data = las_file_analyzer(file_path)
    
    # Check for errors in las_file_analyzer result
    if "error" in las_data:
        return {"error": las_data["error"]}
    
    # Create directory for visualizations in the project directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate a unique filename based on timestamp and visualization type
    timestamp = int(time.time())
    image_filename = f"{visualization_type}_{timestamp}.png"
    image_path = os.path.join(viz_dir, image_filename)
    
    # Extract curve data from las_data
    available_curves = {curve["name"]: curve for curve in las_data["curves_summary"]["curves"]}
    
    # Validate requested curves
    for curve in curve_names:
        if curve not in available_curves:
            return {"error": f"Curve '{curve}' not found in the LAS file. Available curves: {', '.join(available_curves.keys())}"}
    
    # Read the LAS file directly for visualization
    try:
        import lasio
        las = lasio.read(file_path)
        df = las.df()
    except Exception as e:
        return {"error": f"Error reading LAS file for visualization: {str(e)}"}
    
    # Apply depth range filter if specified
    depth_range = options.get("depth_range", None)
    if depth_range and len(depth_range) == 2:
        min_depth, max_depth = depth_range
        df = df[(df.index >= min_depth) & (df.index <= max_depth)]
    
    # Get title or generate default
    title = options.get("title", f"Well Log Visualization - {visualization_type.title()}")
    
    # Generate the appropriate visualization based on type
    if visualization_type.lower() == "multi_track":
        # Multi-track log plot implementation
        track_widths = options.get("track_widths", [1] * len(curve_names))
        colors = options.get("colors", [f"C{i}" for i in range(len(curve_names))])
        
        # Create figure with appropriate dimensions
        fig_width = 2 + 2 * len(curve_names)  # Base width + width per track
        fig_height = 12  # Fixed height for well logs (typically tall)
        fig, axes = plt.subplots(1, len(curve_names), figsize=(fig_width, fig_height), 
                                sharey=True, gridspec_kw={'width_ratios': track_widths})
        
        # Handle single curve case
        if len(curve_names) == 1:
            axes = [axes]
        
        # Plot each curve in its own track
        for i, (curve, ax) in enumerate(zip(curve_names, axes)):
            curve_data = df[curve].values
            depth_data = df.index.values
            
            # Get curve metadata
            curve_info = available_curves[curve]
            unit = curve_info["unit"]
            
            # Plot the curve
            ax.plot(curve_data, depth_data, color=colors[i], linewidth=1.5)
            
            # Set labels and grid
            ax.set_title(curve)
            ax.set_xlabel(f"{curve} ({unit})" if unit else curve)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Invert y-axis (standard for well logs - depth increases downward)
            ax.invert_yaxis()
            
            # Add curve statistics as text
            if curve in las_data["curve_data"]:
                stats = las_data["curve_data"][curve]
                if "min" in stats and "max" in stats and stats["min"] is not None and stats["max"] is not None:
                    ax.text(0.5, 0.02, 
                            f"Min: {stats['min']:.2f}\nMax: {stats['max']:.2f}", 
                            transform=ax.transAxes, ha='center', 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Set common y-label
        fig.text(0.04, 0.5, 'Depth', va='center', rotation='vertical', fontsize=12)
        
        # Set title
        fig.suptitle(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, left=0.1)
        
    elif visualization_type.lower() == "crossplot":
        # Crossplot implementation
        if len(curve_names) < 2:
            return {"error": "Crossplot requires at least two curves"}
        
        # Get x and y curves
        x_curve = options.get("x_curve", curve_names[0])
        y_curve = options.get("y_curve", curve_names[1])
        
        # Validate curves
        if x_curve not in available_curves:
            return {"error": f"X-axis curve '{x_curve}' not found"}
        if y_curve not in available_curves:
            return {"error": f"Y-axis curve '{y_curve}' not found"}
        
        # Get color curve if specified
        color_curve = options.get("color_curve", None)
        if color_curve and color_curve not in available_curves:
            return {"error": f"Color curve '{color_curve}' not found"}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get data
        x_data = df[x_curve].values
        y_data = df[y_curve].values
        
        # Create scatter plot
        if color_curve:
            color_data = df[color_curve].values
            scatter = ax.scatter(x_data, y_data, c=color_data, cmap=options.get("colormap", "viridis"), 
                               alpha=0.7, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f"{color_curve} ({available_curves[color_curve]['unit']})" 
                          if available_curves[color_curve]['unit'] else color_curve)
        else:
            ax.scatter(x_data, y_data, alpha=0.7, edgecolors='none')
        
        # Set labels and title
        x_unit = available_curves[x_curve]["unit"]
        y_unit = available_curves[y_curve]["unit"]
        
        ax.set_xlabel(f"{x_curve} ({x_unit})" if x_unit else x_curve)
        ax.set_ylabel(f"{y_curve} ({y_unit})" if y_unit else y_curve)
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
    elif visualization_type.lower() == "heatmap":
        # Heatmap implementation
        if len(curve_names) < 1:
            return {"error": "Heatmap requires at least one curve"}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepare data matrix
        data_matrix = np.zeros((len(df), len(curve_names)))
        
        for i, curve in enumerate(curve_names):
            data_matrix[:, i] = df[curve].values
            
            # Normalize each column (curve) to 0-1 range for better visualization
            col_min = np.nanmin(data_matrix[:, i])
            col_max = np.nanmax(data_matrix[:, i])
            if col_max > col_min:
                data_matrix[:, i] = (data_matrix[:, i] - col_min) / (col_max - col_min)
        
        # Create heatmap
        colormap = options.get("colormap", "viridis")
        im = ax.imshow(data_matrix, aspect='auto', cmap=colormap, interpolation='none')
        
        # Set y-ticks (depths)
        depth_step = max(1, len(df) // 20)  # Show at most 20 depth labels
        depths = df.index.values
        y_ticks = np.arange(0, len(depths), depth_step)
        y_tick_labels = [f"{depths[i]:.1f}" for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        # Set x-ticks (curve names)
        ax.set_xticks(np.arange(len(curve_names)))
        ax.set_xticklabels(curve_names, rotation=45, ha='right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value')
        
        # Set labels and title
        ax.set_ylabel('Depth')
        ax.set_title(title)
        
    else:
        return {"error": f"Unsupported visualization type: {visualization_type}"}
    
    # Save the figure to file
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Return the image filename and metadata
    return {
        "image_filename": image_filename,  # Just the filename, not the full path
        "visualization_type": visualization_type,
        "curves": curve_names,
        "metadata": {
            "depth_range": [float(df.index.min()), float(df.index.max())] if not df.empty else None,
            "curve_units": {curve: available_curves[curve]["unit"] for curve in curve_names}
        }
    }

if __name__ == "__main__":
    mcp.run()
