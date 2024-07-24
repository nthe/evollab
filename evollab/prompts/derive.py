from ..models import Template


derive_user = """
You are an Instruction Writer that writes instructions for given #Text#.
Please follow the steps below to write an instruction for which the #Text# would be an answer.

Step 1: Please read the "#Text#" carefully and list all the possible methods to formulate instruction.
Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the 
instruction of high quality. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Instruction#. #Instruction# should be 
of proper quality and should be able to answer the "#Text#".

Step 4: Please carefully review the #Instruction# and identify any unreasonable parts. Ensure that the 
#Instruction# of adequate complexity considering the #Text#. Provide the #Finally Rewritten Instruction#
without any explanation.

Please reply strictly in the following format:

Step 1
#Methods List#:

Step 2
#Plan#: 

Step 3
#Instruction#:

Step 4
#Finally Rewritten Instruction#:


Please, try to rewrite instruction even if it led to overly complex version, almost unanswerable by human.
Do not provide any header, footer, reasoning, or additional information, just the rewritten instruction.

#Text#:
{text}
"""

derive_system = "You are an Instruction Writer that writes instructions for given text."

derive = Template(
    system=derive_system,
    user=derive_user,
)
