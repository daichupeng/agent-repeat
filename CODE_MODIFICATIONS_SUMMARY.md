# Code Modifications for Hierarchical Experiment Log Structure

## Overview
Modified the ToolSandbox codebase to automatically save experiment logs to the new hierarchical directory structure during future runs. The code now generates paths in the format:

```
data/agent_name/base_scenario_name/modification/episode_count/[modelname_memory_timestamp]/
```

## Files Modified

### 1. `tool_sandbox/cli/utils.py`

#### Added Import
```python
from tool_sandbox.scenarios.episode_parameterization import (
    ...
    base_scenario_name,  # Added
    ...
)
```

#### New Function: `generate_hierarchical_output_path()`
**Purpose**: Constructs the hierarchical output directory path based on experiment metadata

**Parameters**:
- `base_output_dir`: Base output directory (typically `data/`)
- `agent_name`: Agent type name (e.g., `deepseek`)
- `scenario_name`: Full scenario name (e.g., `modify_contact_with_message_recency_3_distraction_tools`)
- `episodes`: Number of episodes
- `model_name`: Full model name (e.g., `deepseek-v4-flash-0731`)
- `memory_mode`: Memory mode (FULL or NONE)
- `timestamp_str`: Timestamp in YYMMDDHHMMSS format

**Logic**:
1. Extracts base scenario name using `base_scenario_name()` function
2. Calculates modification by removing base name from full scenario name
3. Formats model+memory+timestamp string
4. Constructs hierarchical path

#### Modified Function: `run_scenario_sequence()`
**New Optional Parameters**:
- `agent_name`: Agent type for hierarchical path
- `model_name`: Model name for hierarchical path
- `timestamp_str`: Timestamp in short format
- `base_output_dir`: Base output directory for hierarchical structure

**Behavior**:
- When all 4 new parameters are provided, generates scenario-specific hierarchical output directory
- Uses generated path for saving scenario results
- Falls back to legacy behavior if parameters not provided (backward compatible)

### 2. `tool_sandbox/cli/__init__.py`

#### Modified Function: `run_sandbox()`
**Changes**:
1. Extracts agent and model names from agent/user objects
2. Generates timestamps in two formats:
   - Long format (MM_DD_YYYY_HH_MM_SS): For display and backward compatibility
   - Short format (YYMMDDHHMMSS): For hierarchical path construction
3. Passes new parameters to all `run_scenario_sequence()` calls:
   - Both multiprocessing pool call
   - Sequential execution call
4. Maintains backward compatibility by keeping original `output_directory` for result_summary.json

## Path Generation Examples

| Scenario | Agent | Model | Memory | Episodes | Path |
|----------|-------|-------|--------|----------|------|
| modify_contact_with_message_recency_3_distraction_tools | deepseek | deepseek-v4-flash-0731 | full | 10 | `data/agent_deepseek/modify_contact_with_message_recency/3_distraction_tools/episode_10/deepseek-v4-flash-0731_memory_full_260818231246/` |
| modify_reminder_with_recency_latest_3_distraction_tools | deepseek | deepseek-v4-flash-0731 | none | 10 | `data/agent_deepseek/modify_reminder_with_recency_latest/3_distraction_tools/episode_10/deepseek-v4-flash-0731_memory_none_260818230702/` |
| send_message_with_contact_content_cellular_off | gpt4 | gpt-4-0125-preview | full | 5 | `data/agent_gpt4/send_message_with_contact_content_cellular_off/base/episode_5/gpt-4-0125-preview_memory_full_260820101530/` |

## How It Works

1. **At Startup** (`run_sandbox()`)
   - Extracts agent type and model name
   - Generates short timestamp (YYMMDDHHMMSS)
   - Passes these to scenario execution functions

2. **Per Scenario** (`run_scenario_sequence()`)
   - Receives scenario name and execution metadata
   - Calls `generate_hierarchical_output_path()` with scenario details
   - Generates unique hierarchical path for this scenario
   - Passes path to `run_scenario_episode()` for execution

3. **Trajectory Saving** (`scenario.play_and_evaluate()`)
   - Uses the hierarchical path as the output directory
   - Saves all trajectories and results to the hierarchical structure
   - Each scenario's files organized in its own folder

## Backward Compatibility

- `result_summary.json` still saved to original flat directory for now
- Existing code that doesn't pass new parameters continues to work
- Legacy output directories unchanged
- New functionality only activates when all hierarchical parameters provided

## Testing

Created `test_hierarchical_paths.py` with 4 comprehensive test cases:
- ✓ 3 distraction tools scenario with full memory
- ✓ Different augmentation suffixes (10 vs 3 distraction tools)
- ✓ Scenario with no modification (base case)
- ✓ All path components verified

**Result**: All tests passing ✓

## Benefits

1. **Organization**: Scenarios automatically organized by type and parameters
2. **Scalability**: New scenarios/agents don't require manual organization
3. **Discovery**: Easy to find all experiments for a given scenario and configuration
4. **Consistency**: Automatic adherence to naming conventions
5. **Backward Compatible**: Existing code continues to work unchanged

## Future Work

To fully complete the hierarchization:
1. Consolidate or reorganize `result_summary.json` files
   - Option A: Keep one per scenario in hierarchical structure
   - Option B: Create master manifest mapping scenarios to paths
2. Update data loading tools to read from new hierarchical structure
3. Consider moving legacy flat directories to archive
