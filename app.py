from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from IPython.display import display
from flask import jsonify
from model import run_model

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FILE_PATH"] = ""
app.config["TEXT"] = ""
app.config[
    "TABLE_STYLE"
] = 'style="display: block; border: 1px solid green; height: 600px; overflow-y: scroll" class="tb"'
column_names = ["id", "title", "content", "title&content"]


@app.route("/")
def index():
    # Set The upload HTML template '\templates\index.html'
    return render_template("index.html")


# Get the uploaded files
@app.route("/", methods=["POST"])
def processData():
    df = pd.DataFrame(columns=column_names)
    text = app.config["TEXT"]

    if app.config["FILE_PATH"] != "":
        df = pd.read_csv(app.config["FILE_PATH"], index_col=False)

    if "file-upload" in request.form:
        # get the uploaded file
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], uploaded_file.filename
            )  # set the file path
            app.config["FILE_PATH"] = file_path
            uploaded_file.save(file_path)  # save the file
            df = pd.read_csv(file_path)
    else:
        text = request.form["text"]
        app.config["TEXT"] = text

    return render_template(
        "index.html",
        tables=[
            df.style.set_table_attributes(app.config["TABLE_STYLE"]).to_html(
                classes="data"
            )
        ],
        titles=df.columns.values,
        text=text,
    )


# Run the model
@app.route("/results")
def get_run_model():
    bert_html, tfidf_html = run_model()
    return render_template(
        "results.html", tables=[bert_html, tfidf_html], titles=column_names
    )


if __name__ == "__main__":
    app.run(debug=True)
