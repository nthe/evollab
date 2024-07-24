from ..models import Template


augment_user = """
You are a Text Augmenter that fills in missing information of given #Text#.
Please follow the steps below to write an instruction for which the #Text# would be an answer.

Step 1: Please read the "#Text#" carefully and identify the missing information or entities in the text. If there
are no missing entities, please provide a comprehensive plan to augment the text.

Step 2: Please create a comprehensive plan to fill in the missing information or entities in the #Text#. The plan
should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Augmented Text#. #Augmented Text# can only add 10 to
20 words into the "#Text#".

Step 4: Please carefully review the #Augmented Text# and identify any unreasonable parts. Ensure that the #Augmented
Text# is only a more complex version of the #Text#. Just provide the #Finally Augmented Text# without any explanation.

Please reply strictly in the following format:

Step 1
#Methods List#:

Step 2
#Plan#:

Step 3
#Augmented Text#:

Step 4
#Finally Augmented Text#:


Please, try to rewrite instruction even if it led to overly complex version, almost unanswerable by human.
Do not provide any header, footer, reasoning, or additional information, just the rewritten instruction.

#Text#:
{text}
"""

augment_system = (
    "You are an Text Augmenter that fills in missing information in the text."
)

augment = Template(
    system=augment_system,
    user=augment_user,
)
