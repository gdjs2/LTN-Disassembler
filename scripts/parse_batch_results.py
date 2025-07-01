from pathlib import Path
import ast
import re

def parse_batch_results(file_path):
    """
    Parse the batch_results.txt file into three lists.
    
    Args:
        file_path (str): Path to the batch_results.txt file
        
    Returns:
        tuple: (binaries_list, code_f1_scores, data_f1_scores)
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    
    # Parse binaries (first line) - need to handle PosixPath objects
    binaries_line = lines[0]
    binaries_str = binaries_line.split('Binaires: ')[1]
    
    # Extract path strings using regex instead of eval for safety
    path_pattern = r"PosixPath\('([^']+)'\)"
    paths = re.findall(path_pattern, binaries_str)
    binaries = paths
    
    # Parse Code F1 scores (second line)
    code_f1_line = lines[1]
    code_f1_str = code_f1_line.split('Code F1 scores: ')[1]
    code_f1_scores = ast.literal_eval(code_f1_str)
    
    # Parse Data F1 scores (third line)
    data_f1_line = lines[2]
    data_f1_str = data_f1_line.split('Data F1 scores: ')[1]
    data_f1_scores = ast.literal_eval(data_f1_str)
    
    return binaries, code_f1_scores, data_f1_scores

# Alternative safer approach using eval with restricted globals
def parse_batch_results_safe_eval(file_path):
    """
    Alternative version using eval with restricted globals for safety.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    
    # Parse binaries (first line)
    binaries_line = lines[0]
    binaries_str = binaries_line.split('Binaires: ')[1]
    
    # Create safe environment for eval with only PosixPath available
    safe_dict = {"PosixPath": Path, "__builtins__": {}}
    binaries_raw = eval(binaries_str, safe_dict)
    binaries = [str(path) for path in binaries_raw]
    
    # Parse Code F1 scores (second line)
    code_f1_line = lines[1]
    code_f1_str = code_f1_line.split('Code F1 scores: ')[1]
    code_f1_scores = ast.literal_eval(code_f1_str)
    
    # Parse Data F1 scores (third line)
    data_f1_line = lines[2]
    data_f1_str = data_f1_line.split('Data F1 scores: ')[1]
    data_f1_scores = ast.literal_eval(data_f1_str)
    
    return binaries, code_f1_scores, data_f1_scores

# Usage example
if __name__ == '__main__':
    file_path = '/home/zhaoqi.xiao/Projects/ltn-ghidra/debug/20250629_235236/batch_results.txt'
    
    # Try the regex approach first (safer)
    try:
        binaries, code_f1, data_f1 = parse_batch_results(file_path)
    except Exception as e:
        print(f"Regex approach failed: {e}")
        # Fallback to safe eval approach
        binaries, code_f1, data_f1 = parse_batch_results_safe_eval(file_path)
    
    print(f"Number of binaries: {len(binaries)}")
    print(f"Number of code F1 scores: {len(code_f1)}")
    print(f"Number of data F1 scores: {len(data_f1)}")
    
    # Print first few items as examples
    print(f"\nFirst 3 binaries:")
    for i, binary in enumerate(binaries[:3]):
        print(f"  {i+1}. {Path(binary).name}")
    
    print(f"\nFirst 3 code F1 scores: {code_f1[:3]}")
    print(f"First 3 data F1 scores: {data_f1[:3]}")