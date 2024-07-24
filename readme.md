# EvolLab

Small utility for text augmentation, instruction generation and evolution.

## Usage
The CLI usage:
```sh
Usage: python -m evollab [OPTIONS] COMMAND [ARGS]...

Options:
  -m, --model TEXT          Large language model to use
  -f, --output_format TEXT  Expected output format
  -t, --temperature FLOAT   Diversity of generated output
  --top_p FLOAT             Probability of less probable words in output
  --seed INTEGER            Reuse of seed helps with consistency of output
  --n INTEGER               Number of generations to produce
  --silent                  Display spinner during generation process
  --help                    Show this message and exit.

Commands:
  answer   Answer a question from a provided text.
  augment  Augment, by filling missing info or entities, to provided text.
  derive   Derive an instruction from a provided text.
  evolve   Evolve an instruction using a method.
```

The example commands:
```sh
python -m evollab evolve "How far is the sun?"
``` 
```
echo "How far is the sun?" | python -m evollab evolve
```

## References
The utility is based on instructions and ideas derived from following papers:
 - [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/pdf/2406.00770)
 - [Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning](https://arxiv.org/pdf/2402.04833)
 - [Orca 2: Teaching Small Langauage Models How to Reason](https://arxiv.org/pdf/2311.11045)