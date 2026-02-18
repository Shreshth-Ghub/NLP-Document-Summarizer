"""
Flask Web Application for Document Summarization - Phase 6
BERT TextRank, T5 Abstractive, and ROUGE Evaluation
"""

import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from preprocess import TextPreprocessor
from model import ExtractiveSummarizer, TextSummarizer
from document_parser import DocumentParser
from keywords import KeywordExtractor
from textrank import TextRankSummarizer
from ner import NERExtractor            # NER ENABLED
from multilingual import MultilingualProcessor
from multi_document import MultiDocSummarizer

# Phase 6: ML evaluation
try:
    from evaluation import SummaryEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: evaluation.py not found. ROUGE scores will not be available.")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB for multiple files

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP components
preprocessor = TextPreprocessor()
summarizer = ExtractiveSummarizer()
textrank_summarizer = TextRankSummarizer()
text_summarizer = TextSummarizer()      # Phase 6: unified summarizer
keyword_extractor = KeywordExtractor()
ner_extractor = NERExtractor()          # NER ENABLED
multilingual = MultilingualProcessor()
multi_doc_summarizer = MultiDocSummarizer()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render home page with upload form."""
    languages = multilingual.get_supported_languages()
    return render_template('index.html', languages=languages)


@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle file upload and generate summary."""
    # Check if files are present
    if 'files[]' not in request.files:
        flash('No files uploaded', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    
    # Filter out empty filenames
    files = [f for f in files if f.filename != '']
    
    if not files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    # Check if all files are allowed
    for file in files:
        if not allowed_file(file.filename):
            flash(f'File {file.filename}: Only .txt, .pdf, and .docx files are supported', 'error')
            return redirect(url_for('index'))
    
    try:
        # Parse all documents
        documents = []
        filepaths = []
        
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)
            
            # Parse document
            parser = DocumentParser()
            parsed_doc = parser.parse(filepath)
            documents.append({
                'text': parsed_doc['text'],
                'filename': filename,
                'format': parsed_doc['format'].upper()
            })
        
        # Get parameters from form
        length_mode = request.form.get('length_mode', 'sentences')
        num_sentences = int(request.form.get('num_sentences', 5))
        summary_percentage = int(request.form.get('summary_percentage', 25))
        summary_method = request.form.get('method', 'frequency')
        language = request.form.get('language', 'english')
        
        # Keyword extraction params
        extract_keywords = request.form.get('extract_keywords', 'yes') == 'yes'
        num_keywords = int(request.form.get('num_keywords', 10))
        keyword_method = request.form.get('keyword_method', 'tfidf')
        
        # NER params (now active)
        extract_entities = request.form.get('extract_entities', 'no') == 'yes'
        
        # Multi-document processing
        is_multi_doc = len(documents) > 1
        
        if is_multi_doc:
            # Combine documents
            combined = multi_doc_summarizer.combine_documents(documents)
            text = combined['combined_text']
            doc_stats = multi_doc_summarizer.get_document_stats(documents)
        else:
            text = documents[0]['text']
            doc_stats = None
        
        # Auto-detect language if set to auto
        if language == 'auto':
            language = multilingual.detect_language(text)
        
        # Calculate summary length based on mode
        if length_mode == 'percentage':
            num_sentences = multi_doc_summarizer.calculate_summary_length(
                text,
                percentage=summary_percentage
            )
        else:
            num_sentences = max(1, min(num_sentences, 20))
        
        num_keywords = max(5, min(num_keywords, 20))
        
        # Phase 6: Generate summary using unified TextSummarizer
        if summary_method in ['bert_textrank', 'abstractive']:
            # Use new ML methods (may fall back to TextRank in model.py)
            summary = text_summarizer.summarize(
                text, method=summary_method, num_sentences=num_sentences
            )
            
            # Build summary_data structure for compatibility
            import nltk
            sentences = nltk.sent_tokenize(text)
            summary_sentences = nltk.sent_tokenize(summary)
            
            summary_data = {
                'summary': summary,
                'summary_sentences': summary_sentences,
                'num_sentences': len(summary_sentences),
                'original_sentences': len(sentences),
                'compression_ratio': len(summary_sentences) / len(sentences) if len(sentences) > 0 else 0,
                'method': summary_method,
                'word_count_original': len(text.split()),
                'original_text_length': len(text),
                'summary_length': len(summary)
            }
        elif summary_method == 'textrank':
            summary_data = textrank_summarizer.summarize(text, num_sentences)
            summary_data['num_sentences'] = num_sentences
            summary_data['original_sentences'] = summary_data.get(
                'original_sentences', len(text.split('.'))
            )
            summary_data['word_count_original'] = len(text.split())
            summary_data['original_text_length'] = len(text)
            summary_data['summary_length'] = len(summary_data['summary'])
        else:
            # Use original extractive methods
            summary_data = summarizer.summarize_text(
                text=text,
                num_sentences=num_sentences,
                method=summary_method,
                preprocessor=preprocessor
            )
        
        # Phase 6: Calculate ROUGE scores if evaluation available
        rouge_scores = None
        quality_assessment = None
        
        if EVALUATION_AVAILABLE:
            try:
                evaluator = SummaryEvaluator()
                eval_result = evaluator.evaluate_summary_quality(
                    text, summary_data['summary']
                )
                rouge_scores = eval_result['scores']
                quality_assessment = {
                    'level': eval_result['quality_level'],
                    'color': eval_result['quality_color'],
                    'avg_f1': eval_result['average_f1']
                }
            except Exception as e:
                print(f"ROUGE evaluation error: {e}")
        
        # Phase 6: If abstractive, also generate extractive baseline for comparison
        extractive_summary = None
        comparison_scores = None
        
        if summary_method == 'abstractive' and EVALUATION_AVAILABLE:
            try:
                extractive_summary = text_summarizer.summarize(
                    text,
                    method='textrank',
                    num_sentences=num_sentences
                )
                
                evaluator = SummaryEvaluator()
                comparison = evaluator.compare_summaries(
                    text,
                    extractive_summary,
                    summary_data['summary']
                )
                
                comparison_scores = {
                    'extractive': comparison['summary1_evaluation'],
                    'abstractive': comparison['summary2_evaluation'],
                    'winner': comparison['winner']
                }
            except Exception as e:
                print(f"Comparison error: {e}")
        
        # Extract keywords if requested
        keywords_data = None
        if extract_keywords:
            keyword_extractor.stop_words = multilingual.get_stopwords(language)
            keywords_data = keyword_extractor.extract_keywords(
                text=text,
                method=keyword_method,
                top_n=num_keywords
            )
        
        # Extract entities if requested (NER enabled)
        entities_data = None
        if extract_entities:
            entities_data = ner_extractor.extract_entities(text)
        
        # Clean up uploaded files
        for filepath in filepaths:
            os.remove(filepath)
        
        # Render results
        return render_template(
            'result.html',
            original_text=text,
            summary=summary_data['summary'],
            summary_sentences=summary_data['summary_sentences'],
            num_sentences=summary_data['num_sentences'],
            original_sentences=summary_data['original_sentences'],
            compression_ratio=f"{summary_data['compression_ratio']:.2%}",
            method=summary_data['method'],
            word_count_original=summary_data['word_count_original'],
            original_length=summary_data['original_text_length'],
            summary_length=summary_data['summary_length'],
            file_format=documents[0]['format'] if len(documents) == 1 else 'MULTIPLE',
            filename=documents[0]['filename'] if len(documents) == 1 else f"{len(documents)} documents",
            keywords_data=keywords_data,
            entities_data=entities_data,
            language=language,
            language_name=multilingual.LANGUAGE_NAMES.get(language, 'English'),
            is_multi_doc=is_multi_doc,
            doc_stats=doc_stats,
            documents=documents if is_multi_doc else None,
            length_mode=length_mode,
            summary_percentage=summary_percentage if length_mode == 'percentage' else None,
            # Phase 6 new variables:
            rouge_scores=rouge_scores,
            quality_assessment=quality_assessment,
            extractive_summary=extractive_summary,
            comparison_scores=comparison_scores
        )
    
    except ValueError as e:
        flash(f'Error parsing file: {str(e)}', 'error')
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error processing files: {str(e)}', 'error')
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """Render about page with project information."""
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
