from ..models import Template


introspect_system = """
You are a helpful, precise but picky assistant for checking the quality
of the answer to a given instruction.
""".strip()

introspect_user = """
## Instruction:
{instruction}

## Response:
{response}

We would like you to answer several questions related to the quality
of the answer to the given instruction.
 1. Why this answer is not good for the given instruction? Analysis
    based on the Helpfulness, Relevance, Accuracy, Level of Details,
    and Structure.
 2. Based on the reason you provided, please generate a better
    answer while preserving the same content. To achieve that, you may
    want to adjust the level of details, add bullet points, or use
    comprehensive words, etc.
    
Use the following format template:

## Analysis:
<analysis here>

## Response:
<response here>

""".strip()


introspect = Template(
    system=introspect_system,
    user=introspect_user,
)
