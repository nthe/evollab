from ..models import Template


initial_method = """
You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more 
complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do 
not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the 
#Instruction# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can 
only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the 
#Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten 
Instruction# without any explanation.

Please reply strictly in the following format:

Step 1
#Methods List#:

Step 2
#Plan#: 

Step 3
#Rewritten Instruction#:

Step 4
#Finally Rewritten Instruction#:

"""

evolve_system = "You are an Instruction Rewriter that rewrites the given instruction into a more complex version."

evolve_user = """
{method}

Please, try to rewrite instruction even if it led to overly complex version, almost unanswerable by human.
Do not provide any header, footer, reasoning, or additional information, just the rewritten instruction.

#Instruction#:
{instruction}
"""


evolve = Template(
    system=evolve_system,
    user=evolve_user,
)
