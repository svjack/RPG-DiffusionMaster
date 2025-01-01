import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def DeepSeek(prompt, in_notebook = False):
    from openai import OpenAI
    import re
    with open(".cache/DeepSeek-key.txt", "r") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key,
                    base_url="https://api.deepseek.com")
    #with open('template/template.txt', 'r') as f:
    #    template=f.readlines()
    #user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    #textprompt= f"{' '.join(template)} \n {user_textprompt}"
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": '''
            You are a master of composition who excels at extracting key objects and their attributes from input text and supplementing the original text with more detailed imagination, creating layouts that conform to human aesthetics. Your task is described as follows:

Extract the key entities and their corresponding attributes from the input text, and determine how many regions should be splited.
For each key object identified in the previous step, use precise spatial imagination to assign each object to a specific area within the image and start numbering from 0. The area refers to dividing the entire image into different regions for a general layout. Each key entities is assigned to a region. And for each entity in the region, give it a more detailed description based on the original text. This layout should segment the image and strictly follow the method below:
a. Determine if the image needs to be divided into multiple rows (It should be noted that a single entity should not be split into different rows, except when describing different parts of a person like the head, clothes/body, and lower garment):
• If so, segment the image into several rows and assign an identifier to each row from top to bottom (e.g., Row0, Row1, ...).
• Specify the percentage of height each row occupies within the image (e.g., Row0 (height=0.33) indicates that the row occupies 33% of the height of the entire upper portion of the image).
b. Within each row, further assess the need for division into multiple regions (it should be noted that each region should contain only one entity):
• If required, divide each row from left to right into several blocks and assign a number to each block (e.g., Region0, Region1, ...).
• Specify the percentage of width each block occupies within its respective row (e.g., Region0 (Row0, width=0.5) denotes that the block is located in Row0 and occupies 50% of the width of that row's left side).
c. Output the overall ratio along with the regional prompts:
• First, combine each row's height separated by semicolons like Row0_height; Row1_height; ...; Rown_height. If there is only one row, skip this step.
• Secondly, attach each row's regions' width after each row's height separated with commas, like Row0_height,Row0_region0_width,Row0_region1_width,...Row0_regionm_width;Row1_height,Row1_region0_width,...;Rown_height,...Rown_regionj_width.
• If the row doesn't have more than one region, just continue to the next row.
• It should be noted that we should use decimal representation in the overall ratio, and if there is only one row, just omit the row ratio.
            '''
        },
        {
            "role": "user",
            "content": "A green twintail hair girl wearing a white shirt printed with green apple and wearing a black skirt."
        },
        {
            "role": "assistant",
            "content": '''
Key entities identification:
We only identify a girl with attribute: green hair twintail, red blouse, blue skirt, so we hierarchically split her features from top to down.
1). green hair twintail (head features of the girl)
2). red blouse (clothes and body features of the girl)
3). blue skirt (Lower garment)
So we need to split the image into 3 subregions.
Plan the structure split for the image:
a. Rows
Row0(height=0.33): Top 33% of the image, which is the head of the green twintail hair girl
Row1(height=0.33): Middle 33% part of the image, the body of the girl which is the red blouse part
Row2(height=0.33): Bottom 33% part of the lower body of the girl, which is the blue skirt
There is no need to split each row into different regions, so each row is a subregion
b. Regions with rows:
Region0:(Row0,width=1) Lush green twintails cascade down, framing the girl's face with lively eyes and a subtle smile, accented by a few playful freckles
Region1: (Row1,width=1) A vibrant red blouse, featuring ruffled sleeves and a cinched waist, adorned with delicate pearl buttons, radiates elegance
Region2: (Row2,width=1) pleated blue skirt, knee-length, sways gracefully with each step, its fabric catching the light, paired with a slender white belt for a touch of sophistication.
c. Overall ratio:
Row0_height,(Row0_region0_width only one region in the row, skip); Row1_height,(Row1_region1_width only one region in the row, skip); Row2_height,(Row2_region2_width only one region in the row, skip)
Final split ratio: 1;1;1
Regional Prompt: Lush green twintails cascade down, framing the girl's face with lively eyes and a subtle smile, accented by a few playful freckles BREAK A vibrant red blouse, featuring ruffled sleeves and a cinched waist, adorned with delicate pearl buttons, radiates elegance BREAK pleated blue skirt, knee-length, sways gracefully with each step, its fabric catching the light, paired with a slender white belt for a touch of sophistication.
            '''
        },
        {
            "role": "user",
            "content": "A girl with white ponytail and black dress are chatting with a blonde curly hair girl in a white dress in a cafe."
        },
        {
            "role": "assistant",
            "content": '''
Key entities identification:
We only identify two girls each with two attributes, #girl1(white ponytail,black dress) #girl2(blonde curly hair,blue skirt) so we hierarchically split their features from top to down.
white ponytail (head features of the girl on the left)
black dress (clothes features of the girl on the left)
blonde curly hair (head features of the girl on the right)
blue skirt (clothes of the girl on the right)
Plan the structure split for the image:
So we need to split the image into 4 subregions.
a. Rows
Since we have four key entities, we should split the image into 4 different regions, and two rows, the girls’ head in the top row, the girls’ body in the bottom row
Row0 (height=0.5): Encompasses the heads and upper torsos of both women.
Row1 (height=0.5): Includes the lower torsos of both women, down to where the table cuts across the image.
b. Regions within rows
Region0 (Row0, width=0.5): White ponytail girl, focusing on her sleek, flowing hair and the subtle expression of engagement in her conversation..
Region1 (Row0, width=0.5): Blonde curly hair girl, emphasizing her vibrant curls and the lively sparkle in her eyes as she engages in the chat.
Region2 (Row1, width=0.5): Her elegant black dress, highlighting the fabric's texture and any intricate details, like lace or embroidery.
Region3 (Row1, width=0.5):Her white dress, capturing its flowy silhouette, possibly with floral patterns or delicate folds to accentuate its elegance.
c. Overall ratio:
Row0_height,Row0_region0_width,Row0_region1_width;Row1_height,Row1_region2_width,Row1_region3_wdith
Final split ratio: 1,1,1;1,1,1
Regional Prompt: Captures the woman in the black dress within the top half of the image. BREAK Contains the woman in the white blouse within the top half. BREAK Shows the lower half of the woman in the black dress. BREAK Displays the lower half of the woman in the white blouse.
            '''
        },
        {
            "role": "user",
            "content": "Two girls are chatting in the cafe"
        },
        {
            "role": "assistant",
            "content": '''
Key entities identification:
The caption identifies two key entities without explicit attributes:
Girl 1 (human subject, unspecified attributes)
Girl 2 (human subject, unspecified attributes)
Since no specific attributes are given for either girl, we will need to imagine details for each entity. We will split the image into two regions to represent each girl.
Plan the structure split for the image:
a. Rows
Considering that we have two key entities and no specific attributes to separate vertically, we can choose to have a single row that encompasses both entities:
Row0 (height=1): This row will occupy the entire image, showing both girls chatting in the cafe.
b. Regions within rows
We will divide the row into two regions to represent each girl:
Region0 (Row0, width=0.5): This region will capture Girl 1, who could be imagined as having a casual hairstyle and a comfortable outfit, seated with a cup of coffee, engaged in conversation.
Region1 (Row0, width=0.5): This region will capture Girl 2, perhaps with a different hairstyle for contrast, such as a bun or waves, and a distinct style of clothing, also with a beverage, actively participating in the chat.
c. Overall ratio:
Since there is only one row, we omit the row ratio and directly provide the widths of the regions within the row:
Final split ratio: 0.5,0.5
Regional Prompt: A casually styled Girl 1 with a warm smile, sipping coffee, her attention focused on her friend across the table, the background softly blurred with the ambiance of the cafe. BREAK Girl 2, with her hair up in a loose bun, laughing at a shared joke, her hands wrapped around a steaming mug, the cafe's cozy interior framing their intimate conversation.
            '''
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    stream=True
    )
    json_dict_out = ""
    for chunk in response:
        json_dict_out += chunk.choices[0].delta.content
        if in_notebook:
            from IPython.display import clear_output
            print(json_dict_out, end="", flush=True)
            clear_output(wait = True)
    print(json_dict_out)
    return get_params_dict(json_dict_out)

def GPT4(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

#def local_llm(prompt,version,model_path=None):
def local_llm(prompt,model_path=None):
    if model_path==None:
        model_id = "Llama-2-13b-chat-hf" 
    else:
        model_id=model_path
    print('Using model:',model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = model.generate(**model_input, max_new_tokens=1024)[0]
        output=tokenizer.decode(res, skip_special_tokens=True)
        output = output.replace(textprompt,'')
    return get_params_dict(output)

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        print("Final split ratio:", final_split_ratio)
    else:
        print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        print("Regional Prompt not found.")

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
    return image_region_dict

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"\*\*Final split ratio\*\*: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        print("Final split ratio:", final_split_ratio)
    else:
        print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"\*\*Regional Prompt\*\*: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        print("Regional Prompt not found.")

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
    return image_region_dict
