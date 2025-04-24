import os
import sys
import json
import tempfile
import io
from pathlib import Path
import pandas as pd
from deepeval import assert_test
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_anthropic import ChatAnthropic
from controllers import LASChatController
from models import LASAnalyzerModel
from state import StateManager
class AnthropicModel(DeepEvalBaseLLM):
    def __init__(self, api_key=None, model_name="claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.model_name = model_name
        self._model = None

    def load_model(self):
        if self._model is None:
            self._model = ChatAnthropic(
                api_key=self.api_key,
                model=self.model_name,
                max_tokens=4096
            )
        return self._model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return f"Anthropic {self.model_name}"

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

def test_well_log_agent_with_real_data(test_queries=None):
    if test_queries is None:
        test_queries = [
            "What is the depth interval of this well?",
            "What is the MCG External Temperature of the well from 135 to 4035 ft?",
            "Compare the various neutron porosity between 1000 ft to 2000 ft.",
            "What is the depth range at which we can use the resistivity value reliably?",
            "List down all the depth range where the gamma ray is beyond 100 API gamma ray unit."
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

        results = []
        test_results = []  # For Excel export

        for test_query in test_queries:
            query_result = controller.handle_query(test_query)
            if not query_result or "error" in query_result:
                continue

            tools_called = []
            for tool_msg in query_result.get("tool_messages", []):
                if isinstance(tool_msg, dict):
                    name = tool_msg.get("name", "unknown_tool")
                    content = tool_msg.get("content", "{}")

                    try:
                        content_json = json.loads(content) if isinstance(content, str) else content
                        if name == "las_file_analyzer":
                            tools_called.append(ToolCall(
                                name=name,
                                description="Analyze a LAS file",
                                input_parameters={"file_path": upload_result.get("file_path")},
                                output=content_json
                            ))
                        elif name == "visualize_well_log":
                            tools_called.append(ToolCall(
                                name=name,
                                description="Generate visualizations",
                                input_parameters={
                                    "file_path": upload_result.get("file_path"),
                                    "visualization_type": content_json.get("visualization_type", ""),
                                    "curve_names": content_json.get("curves", [])
                                },
                                output=content_json
                            ))
                        elif name == "execute_las_code":
                            tools_called.append(ToolCall(
                                name=name,
                                description="Execute Python code",
                                input_parameters={
                                    "file_path": upload_result.get("file_path"),
                                    "code": content_json.get("code", "")
                                },
                                output=content_json
                            ))
                    except (json.JSONDecodeError, TypeError):
                        pass

            real_test_case = LLMTestCase(
                input=test_query,
                actual_output=query_result.get("response", ""),
                tools_called=tools_called
            )

            anthropic_eval_model = AnthropicModel()

            stdout_capture = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = stdout_capture

            task_completion_metric = TaskCompletionMetric(
                threshold=0.7,
                model=anthropic_eval_model,
                include_reason=True,
                verbose_mode=True
            )

            try:
                assert_test(real_test_case, [task_completion_metric])
            finally:
                sys.stdout = original_stdout

            verbose_logs = stdout_capture.getvalue()

            user_goal = extract_from_logs(verbose_logs, "User Goal: ", "\n")
            task_outcome = extract_from_logs(verbose_logs, "Task Outcome: ", "\n")
            score = extract_from_logs(verbose_logs, "Score: ", "\n")
            reason = extract_from_logs(verbose_logs, "Reason: ", "\n")
            test_results.append({
                "LAS File": os.path.basename(test_file_path),
                "Query": test_query,
                "User Goal": user_goal,
                "Task Outcome": task_outcome,
                "Score": score,
                "Reason": reason
            })

            results.append({
                "las_file": os.path.basename(test_file_path),
                "query": test_query,
                "response": query_result.get("response", "")
            })

        controller.cleanup_session_files()

        results_dir = Path("/home/harry/Desktop/updated4/well-log-agent-sandbox/test_results")
        results_dir.mkdir(exist_ok=True)



        # Save to JSON
        results_file = results_dir / "task_completion_results.json"

        # Create the new results data
        new_results = {
            "results": results,
            "test_results": test_results,
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
        excel_path = results_dir / "task_completion_results.xlsx"
        save_results_to_excel(test_results, excel_path)

        return {
            "results": results,
            "test_results": test_results,
            "excel_path": str(excel_path),
            "json_path": str(results_file),
            "summary": {
                "total": len(results)
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    os.makedirs("/home/harry/Desktop/updated4/well-log-agent-sandbox/test_results", exist_ok=True)
    result = test_well_log_agent_with_real_data()

    if "error" in result:
        print(f"Test failed with error: {result['error']}")
        sys.exit(1)

    print(f"\nTest results saved to Excel file: {result.get('excel_path')}")
    print(f"Total queries tested: {result['summary']['total']}")

    import time
    time.sleep(5)
    sys.exit(0)
