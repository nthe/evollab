from ..models import Template


analyze_user = """
The following list shows cases where an Instruction evolves into a 
more complex version of an Instruction. 

For each case, stage 0 represents the Instruction in its initial state, and 
each subsequent stage requires an increase in complexity based on 
the previous stage.

Please identify cases that failed to evolve, and provide their case ID, 
reasons, and general optimization constraint that would ensure proper evolution
of any instruction in the future, not just the instructions at hand.
Do not mention stages in the optimization constraints.

Always compare the current stage with the previous stage only.
If all cases have evolved successfully, return an empty list.

Use the following format to provide feedback:
```json
[
    {{
        "case_id": "<Case ID>",
        "reason": "<Reason>",
        "constraint": "<Constraint>",
    }},
    // repeat for other cases
]
```

#Cases#:
{trajectory}

#Feedback#:
"""

analyze_system = (
    "You are an Instruction Evolution Analyst. "
    "Your concise and clear feedback is crucial for the evolution of instructions."
)

analyze = Template(
    system=analyze_system,
    user=analyze_user,
)
