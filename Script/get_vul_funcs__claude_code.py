import json
import anyio
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, Optional
from abc import ABC, abstractmethod

from datasets import Dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, ToolUseBlock


class ProcessingCallback(Protocol):
    """Protocol for processing callbacks."""
    
    @abstractmethod
    def __call__(self, current_index: int, processed_results: List[Dict[str, Any]], **kwargs) -> None:
        """Execute the callback."""
        pass


class SaveCheckpointCallback:
    """Callback to save processing checkpoints."""
    
    def __init__(self, save_file_format: str = "{excel_path}/{write_to}__checkpoints/chpk{n}.json"):
        self.save_file_format = save_file_format
        self.checkpoint_counter = 0
        self.last_checkpoint_index = -1
    
    def __call__(self, current_index: int, processed_results: List[Dict[str, Any]], **kwargs) -> None:
        excel_path = kwargs.get('excel_path', '')
        write_to = kwargs.get('write_to', 'results')
        
        # Create checkpoint directory
        excel_path_obj = Path(excel_path).parent if excel_path else Path.cwd()
        checkpoint_dir = excel_path_obj / f"{write_to}__checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Generate checkpoint filename
        checkpoint_file = checkpoint_dir / f"chpk{self.checkpoint_counter}.json"
        
        # Only save results since last checkpoint (pagination effect)
        start_idx = self.last_checkpoint_index + 1
        end_idx = len(processed_results)
        checkpoint_results = processed_results[start_idx:end_idx]
        
        # Save checkpoint data
        checkpoint_data = {
            "checkpoint_number": self.checkpoint_counter,
            "start_index": start_idx,
            "end_index": end_idx - 1,
            "current_index": current_index,
            "processed_results": checkpoint_results,
            "timestamp": pd.Timestamp.now().isoformat(),
            "batch_size": len(checkpoint_results),
            "total_processed_so_far": len(processed_results)
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        logging.info(f"Checkpoint saved: {checkpoint_file} (batch: {len(checkpoint_results)} CVEs, indices {start_idx}-{end_idx-1})")
        
        # Update tracking
        self.last_checkpoint_index = current_index
        self.checkpoint_counter += 1


async def get_vul_funcs(cve_id: str, prompt: str):
    options = ClaudeCodeOptions(
        max_turns=50,
        system_prompt="You are a security-focused code analyst. ",
        # cwd=Path("/path/to/project"),  # Can be string or Path
        allowed_tools=["Bash", "WebSearch", "WebFetch", "mcp__patchpeek"]
    )
    async for message in tqdm_async(query(prompt=prompt.format(cve_id=cve_id), options=options), leave=False):
        if isinstance(message, AssistantMessage) and isinstance(message.content[0], ToolUseBlock):
            if message.content[0].name == "mcp__patchpeek__VulnerableFunctionSearchFormatter":
                return message.content[0].input

    raise ValueError(f"Could not retrieve structured output for {cve_id}")


async def main(input_excel_file: str, read_from: str, write_to: str = "wiz-java-claude_code", prompt: str = "", 
               callbacks: Optional[List[ProcessingCallback]] = None, checkpoint_frequency: int = 10):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cve_processing.log')
        ]
    )
    
    if callbacks is None:
        callbacks = []
    
    # Read in the data from an excel file
    excel_path = Path(input_excel_file).expanduser()
    df = pd.read_excel(excel_path, sheet_name=read_from)
    
    # Convert the pandas DataFrame to a HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    cve_ids = dataset["cve_id"]
    
    logging.info(f"Processing {len(cve_ids)} CVEs sequentially")
    logging.info(f"Checkpoint frequency: every {checkpoint_frequency} CVEs")

    processed_results: List[Dict[str, Any]] = []
    
    # Process CVEs one at a time
    for i, cve_id in tqdm(enumerate(cve_ids), desc="Processing CVEs"):
        try:
            result = await get_vul_funcs(cve_id, prompt)
            logging.info(f"Successfully processed {cve_id}")
            processed_results.append({"result": result, "error_msg": ""})
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing {cve_id}: {error_msg}")
            processed_results.append({"result": "", "error_msg": error_msg})
        
        # Execute callbacks at specified frequency
        if callbacks and (i + 1) % checkpoint_frequency == 0:
            callback_kwargs = {
                'excel_path': str(excel_path),
                'write_to': write_to,
                'cve_ids': cve_ids
            }
            for callback in callbacks:
                try:
                    callback(i, processed_results, **callback_kwargs)
                except Exception as e:
                    logging.error(f"Callback error: {e}")
    
    # Execute final callbacks after processing is complete
    # Handles the case where the last checkpoint is not saved due to the last batch being less than checkpoint_frequency in size
    if callbacks:
        callback_kwargs = {
            'excel_path': str(excel_path),
            'write_to': write_to,
            'cve_ids': cve_ids
        }
        for callback in callbacks:
            try:
                callback(len(cve_ids) - 1, processed_results, **callback_kwargs)
            except Exception as e:
                logging.error(f"Final callback error: {e}")
    
    successful_count = sum(1 for r in processed_results if r["result"] != "")
    failed_count = sum(1 for r in processed_results if r["error_msg"] != "")
    
    logging.info(f"Processing complete: {successful_count} successful, {failed_count} failed")
    
    # You can store or process the results here
    # For example: save to file, database, etc.
    dataset = dataset.map(lambda _, idx: {"vuln_funcs": json.dumps(processed_results[idx]["result"]), "error_msg": processed_results[idx]["error_msg"]}, with_indices=True)
 
    logging.info(f"Writing results to sheet: {write_to}")
    
    with pd.ExcelWriter(excel_path, mode="a", if_sheet_exists="replace") as writer:
        dataset.to_pandas().to_excel(writer, sheet_name=write_to, index=False)
    
    return processed_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process CVE data with concurrent processing")
    parser.add_argument("--input-excel-file", type=str, required=True,
                        help="Path to the Excel file to read from and write to")
    parser.add_argument("--read-from", type=str, default="raw", 
                        help="Name of the Excel sheet to read from (default: raw)")
    parser.add_argument("--write-to", type=str, default="wiz-claude_code",
                        help="Name of the Excel sheet to write results to. WARNING: This will overwrite/create the sheet with this name in the input Excel file")
    parser.add_argument("--prompt-file", type=str, default=str(Path(__file__).parent.parent / "prompt.xml"),
                        help="Path to the prompt XML file (default: ../prompt.xml)")
    parser.add_argument("--checkpoint-frequency", type=int, default=10,
                        help="How often to save checkpoints (every N CVEs, default: 10)")
    
    args = parser.parse_args()
    
    # Load prompt from file
    prompt = Path(args.prompt_file).read_text()
    
    # Set up callbacks
    callbacks = [SaveCheckpointCallback()]
    
    anyio.run(main, args.input_excel_file, args.read_from, args.write_to, prompt, callbacks, args.checkpoint_frequency)
