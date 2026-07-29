import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

import pdfplumber
import docx

from werkzeug.utils import secure_filename

from fpdf import FPDF

from dotenv import load_dotenv

from mistralai.client import Mistral






load_dotenv()





client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)




app = FastAPI(
    title="ChatMistral MCQ Generator API"
)



UPLOAD_FOLDER = "uploads"

RESULT_FOLDER = "results"



ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx"
}



os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)




def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".",1)[1].lower()
        in ALLOWED_EXTENSIONS
    )



# ==========================
# Extract Text
# ==========================

def extract_text_from_file(file_path):

    extension = file_path.rsplit(".",1)[1].lower()



    if extension == "pdf":

        text = ""

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                text += page.extract_text() or ""

        return text



    elif extension == "docx":

        document = docx.Document(
            file_path
        )


        text = ""

        for para in document.paragraphs:

            text += para.text + "\n"


        return text



    elif extension == "txt":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()



    return ""







def generate_mcqs(text, number):


    prompt = f"""

You are an expert AI teacher.
Create {number} multiple choice questions from the given study material.


Rules:
1. Each question must have exactly 4 options.
2. Mention correct answer.
3. Questions should be exam oriented.
4. Avoid duplicate questions.


Format:

## MCQ

Question:

A)
B)
C)
D)

Correct Answer:


Study Material:{text}

"""


    response = client.chat.complete(

        model="mistral-small-latest",

        messages=[

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.7

    )


    return response.choices[0].message.content







def save_text_file(content, filename):


    path = os.path.join(
        RESULT_FOLDER,
        filename
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)


    return path






def create_pdf(content, filename):


    pdf = FPDF()


    pdf.add_page()


    pdf.set_font(
        "Arial",
        size=12
    )


    for part in content.split("## MCQ"):


        if part.strip():

            pdf.multi_cell(
                0,
                10,
                part
            )



    path = os.path.join(
        RESULT_FOLDER,
        filename
    )


    pdf.output(path)


    return path





@app.post("/generate")
async def generate(

    file: UploadFile = File(...),

    num_questions: int = Form(...)

):


    if not allowed_file(file.filename):

        return {"error": "Only pdf, txt and docx files allowed" }




    filename = secure_filename(file.filename)



    upload_path = os.path.join(

        UPLOAD_FOLDER,
        filename
    )




    with open(
        upload_path,
        "wb"
    ) as f:

        f.write(await file.read())



  

    text = extract_text_from_file(upload_path)



    if not text:

        return {"error":"Text extraction failed"}




    mcqs = generate_mcqs(

        text,
        num_questions)




    txt_filename = (f"mcqs_{filename}.txt" )


    pdf_filename = (f"mcqs_{filename}.pdf" )



    save_text_file(mcqs, txt_filename )



    create_pdf(mcqs, pdf_filename)




    return {

        "message":
        "MCQs generated successfully",


        "txt_download":
        f"/download/{txt_filename}",


        "pdf_download":
        f"/download/{pdf_filename}",


        "mcqs":
        mcqs

    }


@app.get("/download/{filename}")

def download(filename):


    file_path = os.path.join(RESULT_FOLDER,filename )


    return FileResponse(file_path,filename=filename)

