from flask import Flask, render_template,request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from PyPDF2 import PdfReader
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

vectorizer = TfidfVectorizer()

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data 
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        job = request.form['job']
        reader = PdfReader(file)
        page = reader.pages[0]
        resume = page.extract_text()
        vectors = vectorizer.fit_transform([job, resume])
        similarity = cosine_similarity(vectors)
        return str(similarity[0][1])
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)