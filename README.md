# Image Caption Generation

## Project Overview
This project was completed as part of the **Deep learning in the MSc in AI program at the University of Leeds**. The task was to build an image captioning model on a subset of the COCO dataset, experiment with different architectures and decoding strategies, and evaluate the generated captions using multiple metrics.

## Objectives
- **Build an Encoder-Decoder model** where a pretrained CNN encodes images as feature vectors and an RNN decoder generates captions word-by-word.
- **Compare four decoder architectures** (Vanilla RNN vs LSTM, two size configurations each) and select the best-performing one.
- **Experiment with training strategies**: hyperparameter tuning, early stopping, and weight tying.
- **Generate captions** using both greedy decoding and beam search, and compare the two.
- **Evaluate caption quality** using BLEU score and cosine similarity, and discuss the complementary strengths of both metrics.

## Datasets
- The 5029-image COCO subset (`coco_subset_images.zip`) was provided by the University of Leeds for academic purposes and **cannot be shared publicly**.
- The caption annotations (`annotations_trainval2017.zip`) are publicly available and can be downloaded from [cocodataset.org](https://cocodataset.org/#download).
- `coco_subset_meta.csv` maps the 5,029 image IDs to their filenames and metadata — it is included in this repository.

## Methods
- **Encoder**: pretrained ResNet-152 (ImageNet weights), final FC layer removed, frozen during training. Outputs a 2048-dimensional feature vector per image.
- **Decoder**: linear layer → batch normalisation → linear layer → RNN/LSTM → linear projection to vocabulary logits. Word embeddings learned during training.
- **Vocabulary**: ~2,390 tokens built from training and validation captions, excluding words appearing 3 times or fewer.
- **Training**: Cross-Entropy loss on packed sequences, Adam optimiser (`lr=5e-4`, `weight_decay=1e-4`), early stopping (patience=3), up to 40 epochs, batch size 64.
- **Decoding**: greedy (argmax at each step) and beam search (`beam_size=5`).
- **Metrics**: BLEU-2 and BLEU-3 (NLTK, smoothing method 1), and cosine similarity on average word embeddings.

## Results
- **Best architecture**: LSTM decoder with `embed_size=128`, `hidden_size=256` — best generalisation among the four architectures tested.
- **Beam search vs greedy**: beam search improves average BLEU-2 by ~7% and BLEU-3 by ~20%, while cosine similarity remains largely unchanged across both methods.
- **Weight tying**: tying the embedding and output projection weights increases overfitting in this setting, likely due to the architectural constraint it imposes (`embed_size = hidden_size`).
- **Metric comparison**: BLEU captures exact n-gram overlap; cosine similarity captures semantic meaning. Both metrics are complementary and tell different stories about caption quality.

## How to Run
This project runs on **Google Colab**.

1. Clone the repository:
   ```
   git clone https://github.com/ariannacasiraghi/image-caption-generation.git
   ```
2. Upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Place the following files in your Google Drive under `Colab Notebooks/`:
   - `coco_subset_images.zip` *(course-provided, not included)*
   - `annotations_trainval2017.zip` *(download from [cocodataset.org](https://cocodataset.org/#download))*
   - `coco_subset_meta.csv` *(included in this repository)*
4. Mount your Google Drive when prompted and run all cells in order.

Dependencies are pre-installed on Colab. For reference, see `requirements.txt`.
