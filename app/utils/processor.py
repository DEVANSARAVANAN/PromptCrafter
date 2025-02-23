import pandas as pd
from io import StringIO
from app.utils.model import initialize_gemini_client
from google.genai import types

def preProcessing(dataset):
    prompt_template = """
        
        Input:
        Dataset:{dataset}

        Outputs:
        -  Process the Dataset for getting high accuaracy.
        - Output should be in comma seperated form.
        - Output should be processed data of Dataset that given as input.
        - You must return only the processed data no need any code
        - You have to remove the number in category column
        - Example: post_id,post_text,category

        """
    client = initialize_gemini_client()
    formatted_prompt = prompt_template.format(dataset=dataset)
    response =client.models.generate_content(
        model="gemini-2.0-flash",
        contents=formatted_prompt,
        config=types.GenerateContentConfig(
      temperature= 0.1
   ),
    )
    print('='*100,"PREPROCESSING THE DATA",'='*100)
    print(response.text)
    print('='*200)
    result=response.text[27::]
    cleaned_lines = [line for line in result.split("\n") if line.count(",") == 2]
    cleaned_result = "\n".join(cleaned_lines)
    csv_data = StringIO(cleaned_result)
    df = pd.read_csv(csv_data, header=None, names=['post_id', 'post_text', 'category'], on_bad_lines='skip')
    df.to_csv("processed_data.csv", index=False)
    dataset=pd.DataFrame(pd.read_csv('processed_data.csv'),columns=['post_id', 'post_text', 'category'])
    print("*"*100)
    return dataset




