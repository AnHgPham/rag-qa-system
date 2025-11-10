# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_document_qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set up API Key (for Advanced Model)

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Or export directly
export OPENAI_API_KEY="your-api-key-here"
```

### Step 3: Run Full Pipeline

```bash
# This will download data, train both models, and compare them
python main.py --all
```

**Expected time**: ~15 minutes total
- Download: ~1 minute
- Baseline training: ~10 seconds
- Advanced training: ~10 minutes
- Comparison: ~1 minute

### Step 4: Query the System

```bash
# Using advanced model (default)
python main.py --query "What is the capital of France?"

# Using baseline model
python main.py --query "Who invented the telephone?" --model baseline
```

## 📊 What You'll Get

After running the full pipeline, you'll have:

1. **Trained Models**
   - `outputs/models/baseline/tfidf_retriever.pkl` - Baseline TF-IDF model
   - `data/vector_db/` - Advanced embedding model database

2. **Evaluation Results**
   - `outputs/results/baseline/tf-idf_results.json`
   - `outputs/results/advanced/dense_embedding_results.json`
   - `outputs/results/comparison/comparison_data.json`

3. **Reports & Visualizations**
   - `outputs/results/comparison/model_comparison.png` - Performance charts
   - `outputs/results/comparison/comparison_report.md` - Detailed analysis

4. **Dataset**
   - `data/raw/squad/` - SQuAD dataset files

## 🎯 Individual Commands

### Download Dataset Only

```bash
python main.py --download
```

### Train Baseline Model Only

```bash
python main.py --train-baseline
```

### Train Advanced Model Only

```bash
python main.py --train-advanced
```

### Compare Models

```bash
python main.py --compare
```

## 🌐 Web Interface (Optional)

```bash
# Run Streamlit app
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## 🐛 Troubleshooting

### Issue: "Model not found"

**Solution**: Train the model first
```bash
python main.py --train-baseline  # or --train-advanced
```

### Issue: "API key not found"

**Solution**: Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-key"
```

### Issue: "Out of memory"

**Solution**: Reduce batch size in training scripts
- Edit `scripts/advanced/03_train_advanced.py`
- Change `batch_size=32` to `batch_size=16` or `batch_size=8`

### Issue: "Slow training"

**Solution**: This is normal for advanced model
- Baseline: ~10 seconds
- Advanced: ~10 minutes (encoding 18,896 documents)
- Consider using GPU if available

## 📚 Next Steps

1. **Explore the code**
   - Check `src/baseline/tfidf_retriever.py` for baseline implementation
   - Check `src/embedding/dense_retriever.py` for advanced implementation

2. **Read the documentation**
   - `README.md` - Full project documentation
   - `docs/TECHNICAL_REPORT.md` - Technical details and evaluation

3. **Experiment**
   - Try different queries
   - Modify model parameters
   - Add your own documents

4. **Extend**
   - Add web interface
   - Support more file formats
   - Fine-tune on your domain

## 💡 Tips

- **Fast testing**: Use baseline model for quick experiments
- **Best quality**: Use advanced model for production
- **Hybrid**: Use baseline for filtering, advanced for reranking
- **Save costs**: Cache common queries to avoid repeated LLM calls

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Read `docs/TECHNICAL_REPORT.md` for technical details
- Open an issue on GitHub

Happy coding! 🎉
