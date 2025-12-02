"""
Evaluation script to test the RL task with claude-haiku-4-5.
Runs the task multiple times and reports pass rate.
"""

import anthropic
import os
import json
import time
from task import TASK_PROMPT, grade_solution

# Tool definition for the model
TOOLS = [
    {
        "name": "write_file",
        "description": "Write content to a file. Use this to save your solution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to write"
                },
                "content": {
                    "type": "string", 
                    "description": "Content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "run_python",
        "description": "Execute Python code and return the output. Use this to test your solution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
]

def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
            return output if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Execution timed out (30s limit)"
        except Exception as e:
            return f"Execution error: {str(e)}"
        finally:
            os.unlink(f.name)


def run_single_evaluation(client: anthropic.Anthropic, max_steps: int = 15) -> dict:
    """Run a single evaluation of the task."""
    
    messages = [{"role": "user", "content": TASK_PROMPT}]
    solution_code = None
    steps = 0
    
    while steps < max_steps:
        steps += 1
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        # Check if model is done (no tool use)
        if response.stop_reason == "end_turn":
            # Model finished without writing solution
            break
        
        # Process tool calls
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})
        
        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                
                if tool_name == "write_file":
                    filename = tool_input.get("filename", "")
                    content = tool_input.get("content", "")
                    
                    if filename == "solution.py" or filename.endswith("solution.py"):
                        solution_code = content
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Successfully wrote to {filename}"
                    })
                
                elif tool_name == "run_python":
                    code = tool_input.get("code", "")
                    output = execute_python(code)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output
                    })
        
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break
    
    # Grade the solution
    if solution_code:
        result = grade_solution(solution_code)
        result['steps'] = steps
        result['solution_code'] = solution_code
    else:
        result = {
            'passed': False,
            'score': 0.0,
            'feedback': 'No solution.py was written',
            'steps': steps,
            'solution_code': None
        }
    
    return result


def run_evaluation(n_runs: int = 10, api_key: str = None) -> dict:
    """Run evaluation n_runs times and report statistics."""
    
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    results = []
    passed_count = 0
    
    print(f"Running {n_runs} evaluations...")
    print("-" * 50)
    
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        
        try:
            result = run_single_evaluation(client)
            results.append(result)
            
            if result['passed']:
                passed_count += 1
                print(f"  PASSED (score: {result['score']:.2f}, steps: {result['steps']})")
            else:
                print(f"  FAILED (score: {result['score']:.2f}, steps: {result['steps']})")
                if result.get('tests_failed'):
                    for failed in result['tests_failed'][:2]:  # Show first 2 failures
                        print(f"    - {failed[:80]}...")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'passed': False,
                'score': 0.0,
                'feedback': f'Error: {str(e)}',
                'steps': 0
            })
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Calculate statistics
    pass_rate = passed_count / n_runs
    avg_score = sum(r['score'] for r in results) / n_runs
    avg_steps = sum(r['steps'] for r in results) / n_runs
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Pass rate: {pass_rate:.1%} ({passed_count}/{n_runs})")
    print(f"Average score: {avg_score:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    
    # Analyze failure modes
    if passed_count < n_runs:
        print("\nFailure Analysis:")
        failure_reasons = {}
        for r in results:
            if not r['passed'] and 'tests_failed' in r:
                for test in r['tests_failed']:
                    test_name = test.split(':')[0]
                    failure_reasons[test_name] = failure_reasons.get(test_name, 0) + 1
        
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} times")
    
    return {
        'pass_rate': pass_rate,
        'avg_score': avg_score,
        'n_runs': n_runs,
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL task evaluation')
    parser.add_argument('--runs', type=int, default=10, help='Number of evaluation runs')
    parser.add_argument('--api-key', type=str, help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: No API key found!")
        print("Either:")
        print("  1. Set ANTHROPIC_API_KEY environment variable:")
        print("     export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  2. Pass it as argument:")
        print("     python run_eval.py --api-key 'sk-ant-...'")
        exit(1)
    
    print(f"Using API key: {api_key[:10]}...{api_key[-4:]}")
    
    evaluation_results = run_evaluation(args.runs, api_key=api_key)
    
    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        # Remove solution_code from results to keep file small
        save_results = evaluation_results.copy()
        save_results['results'] = [
            {k: v for k, v in r.items() if k != 'solution_code'}
            for r in evaluation_results['results']
        ]
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")

