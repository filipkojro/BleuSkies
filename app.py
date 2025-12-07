
from flask import Flask, render_template, request, send_file
 
import main

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 * 1000


@app.route("/")
def hello_world():
    return render_template(
        "index.html",
    )


@app.route("/upload", methods=['POST'])
def upload():
    
    path_to_result = "./Dockerfile"
    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        
        # here run script
        path_to_result = main.main(uploaded_file.filename)
        
    return send_file(path_to_result, download_name="output_with_categories.las", as_attachment=True)

@app.route("/v1/status")
def status():
    return {"status": "ok"}, 200


