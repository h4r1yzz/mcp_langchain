from mcp.server.fastmcp import FastMCP
import os
import sys
from typing import Dict, List, Any, Union, Optional
import lasio
import numpy as np

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
                if i != index:  
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

if __name__ == "__main__":
    mcp.run()
