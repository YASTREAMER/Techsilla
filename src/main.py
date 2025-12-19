import os
from flask import Flask, render_template, request, jsonify, redirect, url_for

import LLM
from audio import Recording

app = Flask(__name__)

# Directory to save audio files
UPLOAD_FOLDER = "Audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    # Render the HTML file from templates/
    return render_template("homepage.html")


@app.route("/record", methods=["POST"])
def record():
    try:
        # Start recording when the button is clicked
        Recording()
        return jsonify(
            {
                "status": "success",
                "message": "Recording completed and noise reduction applied!",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/submit", methods=["POST"])
def submit_text():
    user_text = request.form.get("text")
    dropdown_value = request.form.get("dropdown")  # Get the dropdown value
    print(
        f"Redirecting to: {url_for('question', user_input=user_text, dropdown_value=dropdown_value)}"
    )

    if user_text and dropdown_value:
        # Redirect to the /question route, passing the data as query parameters
        return redirect(
            url_for("question", user_input=user_text, dropdown_value=dropdown_value)
        )

    return jsonify({"response": "No text or dropdown value received"})


@app.route("/question")
def question():
    try:
        # Retrieve query parameters from the URL
        user_input = request.args.get("user_input")
        dropdown_value = request.args.get("dropdown_value")

        # If either value is None, return an error page (or redirect back to home page)
        if not user_input or not dropdown_value:
            return redirect(
                url_for("index")
            )  # Redirect to the homepage if parameters are missing

        # Generate questions using your LLM model or any other logic
        inputToLLM = f"The name of the user is {user_input} and the area for which to generate questions is {dropdown_value}"
        # questions = LLM.runner(inputToLLM)  # Assuming this returns a list of questions

        # Render the 'idk.html' template and pass data
        questions = ["HELLO"]
        return render_template(
            "idk.html",
            questions=questions,
            user_input=user_input,
            dropdown_value=dropdown_value,
        )

    except Exception as e:
        # Return an error page if something goes wrong
        return render_template("error.html", error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
