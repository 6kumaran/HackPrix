from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for file upload
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle document upload
@app.route('/', methods=['GET', 'POST'])
def upload_document():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If the user does not select a file, browser submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # If the file extension is not allowed
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type. Only .txt and .pdf are allowed.')
        
        # Save the uploaded file
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Read the contents of the uploaded file
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as f:
                document_content = f.read()
                
            # Display the document content to the user
            return render_template('index.html', document_content=document_content)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
