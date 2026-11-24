# gpt2-java

An implementation of GPT-2 in Java using [JFlow](https://github.com/yaznat/jflow-deprecated). Pretrain, fine-tune, or chat with GPT-2. Comes with both pretrained and instruction-tuned weights for GPT-2.

## GPT-2 124M Weights Download

Pretrained and instruction-tuned weights can be downloaded [here](https://drive.google.com/drive/folders/1E3YDTo_vdSzVtZZ_Ms8XM2njDHgYhc_M?usp=sharing).

## Training Information

The provided pretrained weights were trained on approximately **300 million** tokens from the **OpenWebText** dataset.

The provided instruction-tuned weights were trained on a custom dataset composed of entries from a [grammar correction dataset](https://huggingface.co/datasets/agentlans/grammar-correction) and a [recipe dataset](https://github.com/josephrmartinez/recipe-dataset). This dataset is **included** in `data/`.

## Getting Started

This project requires both **Java** and **Python**. The Java code uses a Python bridge to tokenize text with [tiktoken](https://github.com/openai/tiktoken). Follow the steps below to set everything up.

---

### 1. Install Java (17+ recommended)

Check if Java is installed:

```bash
java -version
```

If not, install a JDK from [Oracle](https://www.oracle.com/java/technologies/downloads/), [Adoptium](https://adoptium.net/), through Homebrew, or other.

### 2. Install Python (3.10+ recommended)

Check if Python is installed:

```bash
python3 --version
```

If not, [download](https://www.python.org/downloads/), install through Homebrew, or other.

### 3. Set up a Python Virtual Environment

If you don't already have a venv with Tiktoken installed:

```bash
python3 -m venv .tiktoken-venv
source .tiktoken-venv/bin/activate  # On Windows: .tiktoken-venv\Scripts\activate.bat
pip install tiktoken
```

### 4. Run GPT-2 Within the Python Virtual Environment

Compile Java files:

```bash
javac -d bin $(find src lib -name "*.java")
```

Run:

```bash
source .tiktoken-venv/bin/activate
java -cp bin ChatWithGPT2
```
