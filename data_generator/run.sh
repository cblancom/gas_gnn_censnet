#!/bin/bash

# Define your databases, factors, and w_type
databases=("ng_case_col.xlsx" "SIN_col_63.xlsx")
factors=(0.2 0.25 0.3 0.35)
w_type="lineal"  # Change this to "SOC", "Taylor", or "MPCC" as needed

# Iterate over factors to create tmux sessions
for i in "${!factors[@]}"; do
    factor=${factors[$i]}
    session_name="gen$(($i+1))"
    output_dir="./gen$(($i+1))"
    
    # Construct the command
    command="source ../venv/bin/activate && python3 modify_df.py ./db/${databases[0]} ./db/${databases[1]} $factor $output_dir $w_type"
    
    # Create a new tmux session, run the command, and keep the session open
    tmux new -d -s "$session_name" "$command; bash"
    
    # Optionally, redirect the output and error messages to a log file
    tmux pipe-pane -t "$session_name" -o "cat > ~/tmux_${session_name}.log"
done

