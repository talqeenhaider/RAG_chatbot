import os
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify

from RAG import process_pdf

from RAG.rag import (
    create_retriever,
    create_rag_chain,
    llm
)

app = Flask(__name__)



# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Create uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# ============================================================
# GLOBAL RAG CHAIN
# ============================================================

main_chain = None



# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PDF UPLOAD
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_pdf():

    global main_chain

    try:

        # ----------------------------------------------------
        # Check PDF
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "message": "Please select a PDF file."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "message": "Please select a PDF file."
            }), 400


        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "success": False,
                "message": "Only PDF files are allowed."
            }), 400
        
        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        pdf_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(pdf_path)

        print("PDF saved:", pdf_path)


        # ----------------------------------------------------
        # PDF → Chunks → Embeddings → Chroma
        # ----------------------------------------------------

        vectordb = process_pdf(pdf_path)


        # ----------------------------------------------------
        # Chroma → Retriever
        # ----------------------------------------------------

        retriever = create_retriever(vectordb)


        # ----------------------------------------------------
        # Retriever → RAG Chain
        # ----------------------------------------------------

        main_chain = create_rag_chain(
            retriever,
            llm
        )


        print("RAG chatbot is ready.")


        return jsonify({
            "success": True,
            "message": "PDF uploaded and processed successfully."
        })


    except Exception as e:

        print("UPLOAD ERROR:", e)

        return jsonify({
            "success": False,
            "message": "Something went wrong while processing the PDF."
        }), 500

# ============================================================
# ASK QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    global main_chain


    data = request.get_json()

    question = data.get(
        "question",
        ""
    ).strip()


    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if main_chain is None:

        return jsonify({
            "answer": "Please upload and process a PDF first."
        })


    try:

        # answer = main_chain.invoke(question)
        result = main_chain(question)

        return jsonify({
            "answer": result["answer"],
            "pages": result["pages"]
        })


    except Exception as e:

        print("CHAT ERROR:", e)


        return jsonify({
            "answer": "Sorry, something went wrong."
        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )



