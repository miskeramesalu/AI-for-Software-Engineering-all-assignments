 <!-- report -->
<!-- Part 1 -->
# AI Tools Assignment - Part 1: Theoretical Understanding

## Q1: Differences between TensorFlow and PyTorch

**TensorFlow:**
- Developed by Google, uses static computation graphs (define-and-run)
- Better for production deployment with TensorFlow Serving, Lite, JS
- Strong visualization tools (TensorBoard)
- More verbose but very structured
- Preferred for: Large-scale production systems, mobile apps

**PyTorch:**
- Developed by Facebook, uses dynamic computation graphs (define-by-run)
- More Pythonic and intuitive for research
- Easier debugging and prototyping
- Preferred for: Academic research, computer vision, rapid experimentation

**When to choose:**
- TensorFlow: When you need production deployment, mobile apps, or enterprise systems
- PyTorch: For research projects, when you need flexibility and easy debugging

## Q2: Jupyter Notebooks Use Cases in AI

1. **Exploratory Data Analysis (EDA):**
   - Interactive data visualization and analysis
   - Step-by-step data preprocessing
   - Immediate feedback on data transformations

2. **Model Prototyping and Experimentation:**
   - Quick iteration on different model architectures
   - Visualizing intermediate results and model performance
   - Sharing reproducible research with code and results

## Q3: spaCy vs Basic String Operations

**spaCy provides:**
- Pre-trained models for NER, POS tagging, dependency parsing
- Linguistic intelligence handling complex language patterns
- Efficient tokenization that understands contractions, punctuation
- Multi-language support out of the box
- Custom pipeline components for domain-specific tasks
- Entity linking and similarity detection

**Basic string operations** only handle simple pattern matching and lack linguistic understanding.

## Comparative Analysis: Scikit-learn vs TensorFlow

| Aspect | Scikit-learn | TensorFlow |
|--------|-------------|------------|
| **Target Applications** | Classical ML algorithms | Deep learning, neural networks |
| **Ease of Use** | Very beginner-friendly | Steeper learning curve |
| **Community Support** | Excellent documentation | Large community, extensive resources |
| **Performance** | Good for small-medium datasets | Optimized for large-scale, GPU acceleration |
| **Deployment** | Simple models | Complex deployment options |

 <!-- Part 2 -->
\\Part 2: Practical Implementation
   - // for Task 1: Iris Classification.ipynb and Task1 (screenshots)
   - Task 2: MNIST CNN.ipynb and Task2 (screenshots) 
   - Task 3: NLP Analysis.ipynb and Task3 (screenshots)
   
# Part 3: Ethics & Optimization

## Ethical Considerations

### MNIST Model Biases:
- **Geographic bias**: Handwriting styles vary across cultures and education systems
- **Age bias**: Older vs younger handwriting patterns may not be equally represented
- **Educational bias**: Literacy levels affect how people write digits

### Amazon Reviews Model Biases:
- **Brand bias**: More reviews for popular brands (Apple, Samsung)
- **Language bias**: Only English reviews analyzed
- **Sentiment bias**: Rule-based approach may miss contextual sentiment

### Mitigation Strategies:
1. **TensorFlow Fairness Indicators**: Analyze model performance across different subgroups
2. **Data augmentation**: Add diverse handwriting samples to MNIST
3. **spaCy's rule refinement**: Add domain-specific rules for better sentiment analysis
4. **Multi-language support**: Expand beyond English reviews

## Troubleshooting Challenge

Common issues and solutions:

**Issue 1: Dimension mismatches in TensorFlow**
```python
# Incorrect:
model.add(layers.Dense(64))  # Missing input_dim

# Correct:
model.add(layers.Dense(64, input_shape=(784,)))