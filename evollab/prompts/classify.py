from ..models import Template


classify_user = """
You are Text Classifier that classifies the given text into one of the following classes:
{classes}

Please follow the steps below to classify the given text into one of the classes:

Step 1: Please read the given #Text# carefully and list all the possible classes that the text can be classified into.

Step 2: Please carefully review the #Classes List# generated in Step 1 and identify the most suitable classes for the given text.
Provide the list of classes that the text can be classified into in the final #Final Classes List# section.

Please reply strictly in the following format:

Step 1
#Classes List#:

Step 2
#Final Classes List#:


#Text#:
{text}
"""

classify_system = "You are an Prompt Template Optimizer that creates better versions of prompt templates."

classify = Template(
    system=classify_system,
    user=classify_user,
)
