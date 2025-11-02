import os
import sys
import json
import tempfile
import io
import pandas as pd
from pathlib import Path
from controllers import LASChatController
from client import LASAnalyzerModel
from state import StateManager
from langchain_anthropic import ChatAnthropic

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import assert_test

def extract_from_logs(logs, start_marker, end_marker):
    start_idx = logs.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)

    if end_marker == "\n":
        end_idx = logs.find("\n", start_idx)
        if end_idx == -1:
            return logs[start_idx:].strip()
    else:
        end_idx = logs.find(end_marker, start_idx)
        if end_idx == -1:
            return ""

    return logs[start_idx:end_idx].strip()

def save_results_to_excel(test_results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create a DataFrame from the new results
    new_df = pd.DataFrame(test_results)

    # Check if the file already exists
    if os.path.exists(file_path):
        try:
            # Read existing data
            existing_df = pd.read_excel(file_path)

            # Append new data to existing data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Save the combined data
            combined_df.to_excel(file_path, index=False, engine='openpyxl')
        except Exception:
            # If appending fails, create a new file
            new_df.to_excel(file_path, index=False, engine='openpyxl')
    else:
        # If file doesn't exist, just save the new data
        new_df.to_excel(file_path, index=False, engine='openpyxl')

def test_well_log_agent_tool_correctness():


    test_cases = [
        {
            "query": "What is the depth interval of this well?",
            "expected_tools": ["get_file_metadata"]
        },
        {
            "query": "What is the MCG External Temperature of the well from 135 to 4035 ft?",
            "expected_tools": ["get_file_metadata", "get_ascii_data"]
        },
        {
            "query": "Compare the various neutron porosity between 1000 ft to 2000 ft.",
            "expected_tools": ["get_file_metadata", "get_ascii_data", "get_visualization"]
        },
        {
            "query": "What is the depth range at which we can use the resistivity value reliably?",
            "expected_tools": ["get_file_metadata", "get_ascii_data"]
        },
        {
            "query": "List down all the depth ranges where the gamma ray is beyond 100 API gamma ray unit.",
            "expected_tools": ["get_file_metadata", "get_ascii_data"]
        }
    ]

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return {"error": "Missing ANTHROPIC_API_KEY environment variable"}

    try:
        model_instance = ChatAnthropic(
            api_key=anthropic_api_key,
            model="claude-3-5-haiku-20241022",
            max_tokens=8192
        )

        python_path = sys.executable
        las_model = LASAnalyzerModel(model_instance, python_path)
        app_state = {}
        state_manager = StateManager(app_state)
        controller = LASChatController(las_model, state_manager)

        temp_dir = tempfile.mkdtemp()
        test_file_path = Path("./test_data/1046506109.las")
        if not test_file_path.exists():
            return {"error": "Test file not found"}

        class MockFileStorage:
            def __init__(self, filename, content):
                self.filename = filename
                self._content = content

            def read(self):
                return self._content

        with open(test_file_path, "rb") as f:
            file_content = f.read()

        mock_file = MockFileStorage(test_file_path.name, file_content)
        upload_result = controller.handle_file_upload(mock_file, temp_dir)

        if not upload_result or "error" in upload_result:
            return {"error": "File upload failed"}

        # Use DeepEval to evaluate
        metric = ToolCorrectnessMetric(
            threshold=0.7,
            should_consider_ordering=False,
            should_exact_match=False,
            include_reason=True,
            verbose_mode=True
        )

        results = []
        for test_case_data in test_cases:
            query = test_case_data["query"]

            query_result = controller.handle_query(query)
            if not query_result or "error" in query_result:
                continue

            tools_called = [
                tool_msg.get("name", "unknown_tool")
                for tool_msg in query_result.get("tool_messages", [])
                if isinstance(tool_msg, dict)
            ]

            # Convert tool names to ToolCall objects
            tool_call_objects = [ToolCall(name=tool_name) for tool_name in tools_called]
            expected_tool_objects = [ToolCall(name=tool_name) for tool_name in test_case_data["expected_tools"]]

            llm_test_case = LLMTestCase(
                input=query,
                actual_output=query_result.get("response", ""),
                tools_called=tool_call_objects,
                expected_tools=expected_tool_objects
            )

            stdout_capture = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = stdout_capture

            try:
                assert_test(llm_test_case, [metric])
                score = 1.0
            except AssertionError:
                score = 0.0
            finally:
                sys.stdout = original_stdout

            verbose_logs = stdout_capture.getvalue()
            expected_tools_str = extract_from_logs(verbose_logs, "Expected Tools:\n", "\n \n")
            tools_called_str = extract_from_logs(verbose_logs, "Tools Called:\n", "\n \n")
            reason = extract_from_logs(verbose_logs, "Reason: ", "\n")

            results.append({
                "LAS File": os.path.basename(test_file_path),
                "query": query,
                "expected_tools": test_case_data["expected_tools"],
                "actual_tools": tools_called,
                "score": score,
                "reason": reason,
                "verbose_logs": verbose_logs,
                "expected_tools_formatted": expected_tools_str,
                "tools_called_formatted": tools_called_str
            })



        controller.cleanup_session_files()

        results_dir = Path("/home/harry/Desktop/updated4/well-log-agent-sandbox/test_results")
        results_dir.mkdir(exist_ok=True)

        # Save to JSON
        results_file = results_dir / "tool_correctness_results.json"

        # Extract the LAS file name
        las_file_name = os.path.basename(test_file_path)

        # Create the new results data
        new_results = {
            "results": results,
            "summary": {
                "total": len(results)
            }
        }

        # Check if the file already exists
        if os.path.exists(results_file):
            try:
                # Read existing data and append new run
                with open(results_file, "r") as f:
                    existing_data = json.load(f)

                if not isinstance(existing_data, list):
                    existing_data = [existing_data]

                existing_data.append(new_results)

                with open(results_file, "w") as f:
                    json.dump(existing_data, f, indent=2)
            except Exception:
                # If appending fails, create a new file
                with open(results_file, "w") as f:
                    json.dump([new_results], f, indent=2)
        else:
            # If file doesn't exist, create it
            with open(results_file, "w") as f:
                json.dump([new_results], f, indent=2)

        # Save to Excel
        excel_results = []
        for result in results:
            excel_results.append({
                "LAS File": las_file_name,
                "Query": result["query"],
                "Expected Tools": ", ".join(result["expected_tools"]),
                "Actual Tools": ", ".join(result["actual_tools"]),
                "Score": result["score"],
                "Reason": result.get("reason", ""),
                "Expected Tools (Formatted)": result.get("expected_tools_formatted", ""),
                "Tools Called (Formatted)": result.get("tools_called_formatted", "")
            })

        excel_path = results_dir / "tool_correctness_results.xlsx"
        save_results_to_excel(excel_results, excel_path)


        return {
            "results": results,
            "excel_path": str(excel_path),
            "summary": {
                "total": len(results)
            }
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    os.makedirs("/home/harry/Desktop/updated4/well-log-agent-sandbox/test_results", exist_ok=True)
    result = test_well_log_agent_tool_correctness()

    if "error" in result:
        print(f"Test failed with error: {result['error']}")
        sys.exit(1)

    print(f"\nTest results saved to Excel file: {result.get('excel_path')}")
    print(f"Total queries tested: {result['summary']['total']}")

    import time
    time.sleep(5)
    sys.exit(0)
