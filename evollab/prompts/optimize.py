from ..models import Template


optimize_user = """
I will provide you with the prompt template for evolving of the instructions.

You need to update this template based on the #Feedback# from the evolution
analysis without harming the performance on other cases, and ensure that 
the complexity increase brought by the optimized template is not lower than 
the previous template.

Maintain the same format and structure as the previous template.
Update existing Steps or add new steps if necessary.
Keep the formatting and structure consistent with the previous template.
Make sure the #Finally Rewritten Instruction# is always the last step.

Do NOT try to answer or fill the method prompt below.
Do NOT append any reasoning, explanation, or additional information at the end of the template.
Just return new, better version of method based on the feedback.

Please provide the new version of optimized 'PROMPT TEMPLATE' in the 
'UPDATED TEMPLATE' section below.


=== PROMPT TEMPLATE ===
{method}


=== FEEDBACK ===
{feedback}


=== UPDATED TEMPLATE ===
"""

optimize_system = "You are an Prompt Template Optimizer that creates better versions of prompt templates."

optimize = Template(
    system=optimize_system,
    user=optimize_user,
)
