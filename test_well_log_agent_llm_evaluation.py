import os
import sys
import json
import tempfile
import io
import re
import pandas as pd
from pathlib import Path
from typing import Optional, NamedTuple

from controllers import LASChatController
from client import LASAnalyzerModel
from state import StateManager
from langchain_anthropic import ChatAnthropic

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase



class ElementLists(NamedTuple):
    expected_elements: list[str]
    actual_elements: list[str]
    matching_elements: list[str]


class LLMAnalysis:
    def __init__(
        self,
        model: ChatAnthropic,
        include_reason: bool = False,
        async_mode: bool = True
    ):
        self.model = model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self._cache: dict = {}

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        return f"""
        Instructions:
        You are an expert evaluator. Given two text outputs - `expected_output` and `actual_output` - your task is to extract and compare their **key semantic concepts**.

        Focus on identifying and matching the core ideas, facts, and meaningful components that convey the **underlying meaning** of each response, regardless of exact wording or phrasing.

        Avoid surface-level or lexical comparisons. Instead, determine what each output is **trying to communicate**, and extract the central concepts or elements.

        When comparing both outputs:
        - Only consider elements as matching if they are **explicitly present in both outputs** or are **semantically equivalent** (i.e. they convey the same factual meaning).
        - Do **not** match concepts that are merely related or belong to the same category.
        - Consider **numerical details** as semantically matchable, even if phrased differently.
        - For well log data, pay special attention to depth values, curve names, and statistical measures.

        Ensure the match is based on **meaning**, not just shared words or phrases.

        IMPORTANT CONSTRAINTS:
        - Each matching element MUST correspond to an element in the expected output.
        - The number of matching elements CANNOT exceed the number of elements in the expected output.
        - Only include an element in the matching list if it truly represents the same information in both outputs.
        - Be strict in your matching criteria to ensure accurate evaluation.

        Your response must contain three lists:
        1. A list of the core semantic elements found in `expected_output`.
        2. A list of the core semantic elements found in `actual_output`.
        3. A list of elements that express similar or equivalent meanings across both outputs.

        expected_output:
        {test_case.expected_output}

        actual_output:
        {test_case.actual_output}

        Respond strictly in this JSON format only.
        **Important**: Ensure the JSON format is strictly valid. Do not include trailing commas or extra punctuation. Do not wrap the output in code blocks.
        {{
            "expected_elements": ["element1", "element2", ...],
            "actual_elements": ["element1", "element2", ...],
            "matching_elements": ["element1", "element2", ...]
        }}
        """

    def process_response(self, test_case: LLMTestCase) -> ElementLists:
        cache_key = (test_case.expected_output, test_case.actual_output)

        # Ensure that LLM only runs once
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(test_case)
        response = self.model.invoke(prompt)
        output = self.clean_json_response(response.content.strip())

        # Parse JSON response
        result = json.loads(output)

        # Trust the LLM's judgment on matching elements
        llm_output = ElementLists(
            expected_elements=result["expected_elements"],
            actual_elements=result["actual_elements"],
            matching_elements=result["matching_elements"]
        )

        self._cache[cache_key] = llm_output
        return llm_output

    def clean_json_response(self, output: str) -> str:
        if output.startswith("```"):
            output = re.sub(r"^```(?:json)?\s*", "", output)
            output = re.sub(r"\s*```$", "", output)

        output = re.sub(r",(\s*[\]}])", r"\1", output)

        return output


class BaseResponseMetric(BaseMetric):
    metric_name: str = ""

    def __init__(
        self,
        model: ChatAnthropic,
        threshold: float = 0.7,
        include_reason: bool = False,
        async_mode: bool = True,
        evaluator: Optional[LLMAnalysis] = None
    ):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.evaluator = evaluator or LLMAnalysis(model, include_reason, async_mode)

        self.score = 0.0
        self.reason = ""
        self.error = None
        self.success = False
        self.llm_output = None

    def calculate_score(self, llm_output: ElementLists) -> float:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.llm_output = self.evaluator.process_response(test_case)
            self.score = self.calculate_score(self.llm_output)
            self.success = self.score >= self.threshold

            if self.include_reason:
                self._generate_reason()

            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def _generate_reason(self):
        if not self.llm_output:
            self.reason = "No LLM output available"
            return

        expected_count = len(self.llm_output.expected_elements)
        actual_count = len(self.llm_output.actual_elements)
        matching_count = len(self.llm_output.matching_elements)

        precision = matching_count / actual_count if actual_count > 0 else 0
        recall = matching_count / expected_count if expected_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        self.reason = (
            f"{self.metric_name} Score: {self.score:.2f}\n"
            f"Expected Elements: {expected_count}\n"
            f"Actual Elements: {actual_count}\n"
            f"Matching Elements: {matching_count}\n"
            f"Precision: {precision:.2f}\n"
            f"Recall: {recall:.2f}\n"
            f"F1: {f1:.2f}\n"
            f"{'PASS' if self.success else 'FAIL'} [threshold={self.threshold:.2f}]"
        )

    def is_successful(self) -> bool:
        return self.success if self.error is None else False

    @property
    def __name__(self):
        return self.metric_name


class ResponsePrecisionMetric(BaseResponseMetric):
    metric_name = "Precision"

    def calculate_score(self, llm_output: ElementLists) -> float:
        if not llm_output.actual_elements:
            return 0.0
        precision = len(llm_output.matching_elements) / len(llm_output.actual_elements)
        return precision


class ResponseRecallMetric(BaseResponseMetric):
    metric_name = "Recall"

    def calculate_score(self, llm_output: ElementLists) -> float:
        if not llm_output.expected_elements:
            return 0.0
        recall = len(llm_output.matching_elements) / len(llm_output.expected_elements)
        return recall


class ResponseF1Metric(BaseResponseMetric):
    metric_name = "F1"

    def calculate_score(self, llm_output: ElementLists) -> float:
        precision = ResponsePrecisionMetric.calculate_score(self, llm_output)
        recall = ResponseRecallMetric.calculate_score(self, llm_output)

        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


def extract_from_logs(logs, start_marker, end_marker):
    start_idx = logs.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    if end_marker == '\n':
        end_idx = logs.find('\n', start_idx)
        return logs[start_idx:end_idx].strip() if end_idx != -1 else logs[start_idx:].strip()
    else:
        end_idx = logs.find(end_marker, start_idx)
        return logs[start_idx:end_idx].strip() if end_idx != -1 else ""


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


def test_well_log_agent_llm_evaluation():
    test_cases = [
        {"query": "What is the depth interval of this well?",
         "expected_output": "Start Depth: 0.0\nEnd Depth: 4411.0\nDepth Interval: 0.5"},
        {"query": "What is the MCG External Temperature of the well from 135 to 4035 ft?",
         "expected_output": "Minimum: 87.85\nMaximum: 115.28\nAverage: 101.23\nStandard Deviation: 7.20\nValid data points: 7801 out of 7801 in the specified range"},
        {"query": "Compare the various neutron porosity between 1000 ft to 2000 ft.",
         "expected_output": "NPRL: Limestone Neutron Por.\nNPOR: Base Neutron Porosity\nNPRS: Sandstone Neutron Por.\nNPRD: Dolomite Neutron Por."},
        {"query": "What is the depth range at which we can use the resistivity value reliably?",
         "expected_output": "Reliable depth range for RTAO: 304.50 - 4404.50 meters\nReliable depth range for R40O: 304.50 - 4404.50 meters\nReliable depth range for R60O: 304.50 - 4404.50 meters"},
        {"query": "List down all the depth ranges where the gamma ray is beyond 100 API gamma ray unit.",
         "expected_output": "1. 90.00 ft to 94.00 ft\n2. 175.50 ft to 175.50 ft\n3. 181.00 ft to 182.00 ft\n4. 183.50 ft to 187.00 ft\n5. 189.00 ft to 190.00 ft\n... and 233 more ranges\nTotal: 238 zones covering 987.00 ft"},
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
        state_manager = StateManager({})
        controller = LASChatController(las_model, state_manager)

        temp_dir = tempfile.mkdtemp()
        las_file = Path("./test_data/1046506112.las")
        with open(las_file, "rb") as f:
            content = f.read()
        class MockFileStorage:
            def __init__(self, filename, content):
                self.filename = filename
                self._content = content
            def read(self): return self._content
        upload = controller.handle_file_upload(MockFileStorage(las_file.name, content), temp_dir)
        if not upload or "error" in upload:
            return {"error": "File upload failed"}

        # Create an LLM evaluator
        evaluator = LLMAnalysis(model_instance, include_reason=True)

        # Create metrics
        precision_metric = ResponsePrecisionMetric(
            model=model_instance,
            threshold=0.7,
            include_reason=True,
            evaluator=evaluator
        )

        recall_metric = ResponseRecallMetric(
            model=model_instance,
            threshold=0.7,
            include_reason=True,
            evaluator=evaluator
        )

        f1_metric = ResponseF1Metric(
            model=model_instance,
            threshold=0.7,
            include_reason=True,
            evaluator=evaluator
        )

        results = []
        for data in test_cases:
            resp = controller.handle_query(data["query"])
            llm_case = LLMTestCase(
                input=data["query"],
                actual_output=resp.get("response", ""),
                expected_output=data["expected_output"]
            )

            # Calculate scores
            precision_score = precision_metric.measure(llm_case)
            recall_score = recall_metric.measure(llm_case)
            f1_score = f1_metric.measure(llm_case)

            # Get the element lists for detailed analysis
            elements = evaluator._cache.get((data["expected_output"], resp.get("response", "")))

            # capture reason, logs, etc.
            results.append({
                "LAS File": os.path.basename(las_file),
                "query": data["query"],
                "expected_output": data["expected_output"],
                "actual_output": resp.get("response", ""),
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
                "reason": f1_metric.reason,
                "expected_elements": elements.expected_elements if elements else [],
                "actual_elements": elements.actual_elements if elements else [],
                "matching_elements": elements.matching_elements if elements else [],
            })

        controller.cleanup_session_files()

        # Save results to JSON and Excel
        results_dir = Path("/home/harry/Desktop/updated4/well-log-agent-sandbox/test_results")
        results_dir.mkdir(exist_ok=True)

        # Save to JSON
        results_file = results_dir / "llm_evaluation_results.json"

        # Extract the LAS file name
        las_file_name = os.path.basename(las_file)

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
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1"],
                "Expected Elements Count": len(result["expected_elements"]),
                "Actual Elements Count": len(result["actual_elements"]),
                "Matching Elements Count": len(result["matching_elements"])
            })

        excel_path = results_dir / "llm_evaluation_results.xlsx"
        save_results_to_excel(excel_results, excel_path)

        return {
            "results": results,
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
    result = test_well_log_agent_llm_evaluation()

    if "error" in result:
        print(f"Test failed with error: {result['error']}")
        sys.exit(1)

    print(f"\nTest results saved to Excel file: {result.get('excel_path')}")
    print(f"Test results saved to JSON file: {result.get('json_path')}")
    print(f"Total queries tested: {result['summary']['total']}")

    import time
    time.sleep(5)
    sys.exit(0)
