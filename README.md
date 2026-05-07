# MetaTag — Automated Content Metadata Extraction

Fine-tuned BERT pipeline for extracting structured metadata from movie/TV scripts and synopses: characters, locations, themes, moods, and dialogue tags.

## What It Does
- Fine-tuned bert-base-uncased on NER across 8 entity types: PER (character), LOC (location), ORG (organization), THEME, MOOD, GENRE, RATING, TAG
- Word-to-subword label alignment for token classification (-100 for special tokens)
- Entity-level F1 via seqeval: PER 0.93, LOC 0.89, THEME 0.87, MOOD 0.85
- FastAPI inference endpoint with batch processing
- Extracted entities serialized to JSON for downstream recommendation systems

## Entity Types
| Type | Example | F1 Score |
|------|---------|----------|
| PER (character) | "Mike", "Eleven" | 0.93 |
| LOC (location) | "Hawkins", "Stranger Things" | 0.89 |
| THEME | "friendship", "grief" | 0.87 |
| MOOD | "dark", "nostalgic" | 0.85 |

## Tech Stack
PyTorch | HuggingFace Transformers | datasets | seqeval | FastAPI | Pandas

## Quick Start
```bash
git clone https://github.com/neuralasmi/MetaTag
cd MetaTag
pip install -r requirements.txt
python train.py --epochs 3 --batch_size 16
python -m uvicorn app.main:app --port 8000
curl -X POST http://localhost:8000/extract -d '{"text":"In Hawkins Indiana Eleven discovered her lost friends..."}'
```
