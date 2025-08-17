from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import subprocess
import sys
from openai import OpenAI
import os
import csv
import re

import uvicorn

# client = genai.Client(api_key=)

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])



def clean_code_blocks(code_text: str) -> str:
    """Remove markdown code fences (```something ... ```)."""
    # Regex to remove any ```<lang> and ending ```
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, code_text.strip())
    if match:
        return match.group(1)
    return code_text.strip()

def clean_non_utf8(file_path: str):
    import chardet
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
    text = raw_data.decode(encoding, errors="replace")
    replacements = {
    "–": "-",  # en dash
    "—": "-",  # em dash
    "�": "",   # unknown char
    "“": '"',  # fancy quotes
    "”": '"',
    "‘": "'",
    "’": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)



def run_python_file(file_path: str):
    """Run a Python file and return stdout, stderr."""
    result = subprocess.run(
        [sys.executable, file_path],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr

def generate_code(task: str) -> str:

    task_breakdown_file = os.path.join('prompts', 'task_breakdowngpt2.txt')
    with open(task_breakdown_file, 'r') as file:
        task_breakdown_prompt = file.read()
    
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=[task, task_breakdown_prompt],
    #     )
    
    prompt = [
        {
            'role': 'system',
            'content': task_breakdown_prompt
        },
        {
            'role': 'user',
            'content': task
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        response_format={ "type": "text" }
    )
    code_text = response.choices[0].message.content

    
    #code_text = response.text.strip()
    
    code_text = clean_code_blocks(code_text)

    return code_text

def fix_code(original_code: str, error_msg: str, task: str) -> str:
    """Ask Gemini to fix Python code given the error."""
    task_breakdown_file = os.path.join('prompts', 'task_breakdowngpt2.txt')
    with open(task_breakdown_file, 'r') as file:
        task_breakdown_prompt = file.read()
    
    fix_prompt = f"""
    You previously generated a python code that followed these rules:
    {task_breakdown_prompt}

    And the python code had to achivve this task:
    {task}

    The Python code u made was:
    ```
    {original_code}
    ```
    The error was:
    ```
    {error_msg}
    ```
    Rewrite the full code so it runs without errors and keeps the same logic.
    Output only the fixed Python code. If the a module is missing, then try to write the code without using 
    it.
    """
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=[fix_prompt],
    # )

    prompt = [
        {
            'role': 'user',
            'content': fix_prompt
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=prompt,
        response_format={ "type": "text" },
        temperature=0
    )
    code_text = response.choices[0].message.content


    #code_text = response.text.strip()
    
    code_text = clean_code_blocks(code_text)

    return code_text
    

@app.get("/")
async def root():
    return {"message": "Welcome to the CSV Upload API"}

@app.post("/api")
async def upload_file(file: UploadFile = File(None),csv_file: UploadFile = File(None)):
    try:
        contents = await file.read()
        text = contents.decode('utf-8')

        if csv_file is not None:
            import pandas as pd
            # Save CSV to disk
            csv_path = os.path.join("uploads", csv_file.filename)
            os.makedirs("uploads", exist_ok=True)
            with open(csv_path, "wb") as f:
                f.write(await csv_file.read())

            # Preview first 5 rows with pandas
            df_preview = pd.read_csv(csv_path, nrows=5)
            preview_csv = df_preview.to_csv(index=False)
            text += f"""
            The uploaded CSV file is saved at: {csv_path}.
            Here are the first 5 rows of the data:
            {preview_csv}
            """
            
        
        code = generate_code(text)

        max_retries = 3
        for attempt in range(max_retries):
            with open('gen_code.py', 'w') as f:
                f.write(code)
            clean_non_utf8('gen_code.py')
            output, error = run_python_file('gen_code.py')

            if error == "":
                return output
            else:
                code = fix_code(code, error, text)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)