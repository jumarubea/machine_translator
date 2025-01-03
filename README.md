# Machine Translator

This repository contains a machine translation system. It uses word embeddings to map English and French words into a shared vector space, enabling translation by finding an optimal transformation matrix \( R \) through gradient descent. The translation process leverages cross-lingual principles for embedding alignment.

## Features

- Utilizes **English** and **French** word embeddings for translation.
- Implements gradient descent to optimize the transformation matrix \( R \).
- Supports alignment and translation of embeddings.
- Includes full training and usage guidance in a Jupyter notebook (`main.ipynb`).
- Built with common Python libraries: `numpy`, `pickle`, `pandas`, `scipy`, and `matplotlib`.

## File Structure

```

project/
│
├── data/ # Directory for datasets and embeddings.
├── utils.py # Utility functions for embedding processing.
├── translator.py # Core translation logic and functions.
├── main.ipynb # Step-by-step training guide and demonstration.
└── model.ipynb # Optional additional exploration of models.

```

## Requirements

Ensure you have Python 3.8+ installed along with the following packages:

```bash
numpy
pickle
pandas
scipy
matplotlib
```

You also need the following resources:

- Pre-trained embeddings:
  - English: `GoogleNews-vectors-negative300.bin.gz`
  - French: Referenced from [vjstark/crosslingual_text_classification](https://github.com/vjstark/crosslingual_text_classification)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jumarubea/machine_translator.git
   cd machine_translator
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained embeddings in the `data/` directory.

## Usage

### Training the Model

1. Open `main.ipynb` in a Jupyter notebook environment.
2. Follow the guided steps to:
   - Load and preprocess word embeddings.
   - Compute the transformation matrix \( R \) using gradient descent.
   - Evaluate the translation accuracy.

### Translating Words

Use the `translator.py` script to translate individual words:

```python
from translator import translate

# put the customer word
e_word = 'weight'
translate(weight, e_word)
```

English: weight --> French: poids

### Utilities

The `utils.py` file cont
ains helper functions for:

- Loading and preprocessing embeddings.
- Computing evaluation metrics.

## How It Works

1. **Embedding Alignment**: Maps English and French embeddings to a shared vector space.
2. **Gradient Descent**: Optimizes the transformation matrix \( R \) that aligns the embeddings.
3. **Translation**: Uses \( f = eR \) to translate an English word embedding (\( e \)) to French (\( f \)).

## Visualization

- Visualizations for alignment and embedding distributions are included in `main.ipynb`.
- Use `matplotlib` to explore embedding relationships.

## Acknowledgements

- Word embeddings from [GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/).
- Preprocessing inspiration from [vjstark/crosslingual_text_classification](https://github.com/vjstark/crosslingual_text_classification).
- Coursera

## License

This project is licensed under the MIT License.

---

Happy translating! 🚀
=======
# machine_translator
