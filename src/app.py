import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm, trange
import torch.nn.functional as F
import streamlit as st
import boto3
import io

AWS_BUCKET_NAME = '4geeks-lyric-generator'
AWS_ACCESS_KEY_ID = 'AKIA43DLKKSK2OPY3DRZ'
AWS_SECRET_ACCESS_KEY = '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'
MODEL_DIR_PATH = '../models/'
GPT_2_TYPE = 'gpt2-medium'
MODEL_FILE_NAME = 'model_'+ GPT_2_TYPE +'.pt'

#Importo tokenizador y modelo
tokenizer = GPT2Tokenizer.from_pretrained(GPT_2_TYPE)

s3 = boto3.client('s3',
                  aws_access_key_id = AWS_ACCESS_KEY_ID,
                  aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
                  
obj = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=MODEL_FILE_NAME)

#Cargo el modelo "fine tuneado"
device=torch.device('cpu')

model = torch.load(io.BytesIO(obj['Body'].read()), map_location=device)
model.to('cpu')

def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=500, #maximum number of words
    top_p=0.8,
    temperature=1.,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list

def generateStreamlit(seed):
	generated = generate(model, tokenizer, seed, entry_count=1)
	generated_transformed = ' '.join(generated)
	to_remove = generated_transformed.split('.')[-1]
	generated_transformed = generated_transformed.replace('<|endoftext|>','')
	generated_transformed = generated_transformed.replace(to_remove,'')
	generated_transformed = generated_transformed.replace('\n',' ')
	generated_transformed = generated_transformed.replace('.',' \n')
	generated_transformed = generated_transformed.replace('?','? \n')
	generated_transformed = generated_transformed.replace('!','! \n')
	generated_transformed = generated_transformed.replace(',',', \n')
	generated_transformed = generated_transformed.replace(')',') \n')
	generatedLyrics = generated_transformed.replace('}','} \n')
	return generatedLyrics

st.write(""" # Bienvenidos a 4geeks Lyrics Generator! """)
input_lyrics = st.text_input('Escribe tu introducción a la letra de una canción','')

gen_button = st.button("Generar",key=1)

if gen_button:
    output_lyrics = generateStreamlit(input_lyrics)
    print(output_lyrics)
    st.write(""" ## La letra generada es:\n""")
    for line in output_lyrics.split('\n'):
        st.write(line)
