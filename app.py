import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from preprocess import TextPreprocessor
from model import ExtractiveSummarizer
from document_parser import DocumentParser
from keywords import KeywordExtractor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP components
preprocessor = TextPreprocessor()
summarizer = ExtractiveSummarizer()
parser = DocumentParser()
keyword_extractor = KeywordExtractor()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render home page with upload form."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle file upload and generate summary."""
    # Check if file is present
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Only .txt, .pdf, and .docx files are supported', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Parse document (works for TXT, PDF, DOCX)
        parsed_doc = parser.parse(filepath)
        text = parsed_doc['text']
        file_format = parsed_doc['format'].upper()
        
        # Get parameters from form
        num_sentences = int(request.form.get('num_sentences', 5))
        summary_method = request.form.get('method', 'frequency')
        extract_keywords = request.form.get('extract_keywords', 'yes') == 'yes'
        num_keywords = int(request.form.get('num_keywords', 10))
        keyword_method = request.form.get('keyword_method', 'tfidf')
        
        # Validate parameters
        num_sentences = max(1, min(num_sentences, 20))
        num_keywords = max(5, min(num_keywords, 20))
        
        # Generate summary
        summary_data = summarizer.summarize_text(
            text=text,
            num_sentences=num_sentences,
            method=summary_method,
            preprocessor=preprocessor
        )
        
        # Extract keywords if requested
        keywords_data = None
        if extract_keywords:
            keywords_data = keyword_extractor.extract_keywords(
                text=text,
                method=keyword_method,
                top_n=num_keywords
            )
        
        # Clean up uploaded file
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
            file_format=file_format,
            filename=filename,
            keywords_data=keywords_data
        )
    
    except ValueError as e:
        flash(f'Error parsing file: {str(e)}', 'error')
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """Render about page with project information."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)