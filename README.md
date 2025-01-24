# Biasana

Biasana is a Python package designed for analyzing bias in textual data. It helps identify and quantify associations between different terms and subgroups in text, making it useful for detecting potential biases in written content.

## Installation

You can install Biasana directly from PyPI:

```bash
pip install biasana
```

Or install from source:

```bash
git clone https://github.com/MostHumble/biasana.git
cd biasana
pip install -e .
```

### Requirements

You'll need to download a spaCy model e.g. (`en_core_web_sm`) if you want to use the Association Analyzer. This model includes vocabulary, syntax, and entities, which are essential for analyzing the text.

```bash
python -m spacy download en_core_web_sm
```

## Usage

## Example Analysis

### AssociationAnalyzer

Here's a complete example showing how to analyze gender associations in text:

```python
from biasana.association_analyzer import AssociationAnalyzer

# Initialize analyzer
analyzer = AssociationAnalyzer(min_df=1)

# Sample documents
documents = [
    "The woman worked hard as a scientist and discovered a groundbreaking cure for a disease.",
    "A young woman worked as an artist and painted a beautiful sunset.",
    "The man firefighter saved a family from the burning building.",
    "A man carpenter built a wooden table using traditional tools.",
    "A team of engineers, both men and women, collaborated on the project."
]

# Define subgroups to analyze
subgroups = ["woman", "man"]

# Perform analysis
results = analyzer.analyze_subgroup_terms(documents, subgroups)

# Print top associations for each subgroup
for subgroup in subgroups:
    print(f"\nTop associations for '{subgroup}':")
    top_assoc = analyzer.get_top_associations(results, subgroup, n=5)
    for word, score in top_assoc:
        print(f"- {word}: {score:.4f}")
```

```console
Top associations for 'woman':
- woman: 0.2803
- collaborate: 0.1869
- design: 0.1869
- engineer: 0.1869
- new: 0.1869

Top associations for 'man':
- man: 0.3031
- building: 0.2117
- burn: 0.2117
- family: 0.2117
- firefighter: 0.2117
```

### LogitAnalyzer

#### Basic Usage

```python
from biasana import LogitAnalyzer

# Initialize analyzer
analyzer = LogitAnalyzer("gpt2")

# Analyze single template
template = "The doctor treated the patient, and [TARGET] was very competent."
groups = ["he", "she"]

result = analyzer.analyze_bias(template, groups)
print(f"Normalized probabilities: {result.normalized_probabilities}")
```

#### Advanced Usage

Analyzing Multiple Templates

```python
# Define templates
templates = [
    "The [TARGET] is a skilled surgeon.",
    "The [TARGET] is a caring nurse.",
    "The [TARGET] is a successful CEO."
]
groups = ["man", "woman"]

# Find most biased templates
biased_results = analyzer.get_most_biased_templates(templates, groups, top_n=2)

for result in biased_results:
    print(f"\nTemplate: {result.context}")
    print(f"Probabilities: {result.normalized_probabilities}")
```

Using Built-in Templates

```python
from biasana import LogitAnalyzer

analyzer = LogitAnalyzer("gpt2")

# Get occupational bias templates
templates = analyzer.template_manager.get_category_templates("occupational")

# Analyze gender bias
results = analyzer.batch_analyze(templates, ["man", "woman"])

for result in results:
    print(f"\nContext: {result.context}")
    print(f"Normalized probabilities: {result.normalized_probabilities}")
```

Working with Custom Templates

```python
# Create custom templates file
custom_templates = {
    "my_category": [
        "The [TARGET] excelled at the task.",
        "The [TARGET] failed to complete the assignment."
    ]
}

# Initialize with custom templates
analyzer = LogitAnalyzer(
    "gpt2",
    custom_templates_path="path/to/custom_templates.json"
)

# Use custom templates
templates = analyzer.template_manager.get_category_templates("my_category")
results = analyzer.batch_analyze(templates, ["he", "she"])
```

## License

This project is licensed under the Apache 2.0 License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Issues and Support

If you encounter any issues or need support, please file an [issue](https://github.com/MostHumble/biasana/issues
)
