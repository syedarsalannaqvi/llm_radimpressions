import os
import sys
import pandas as pd
import torch
import time
from tqdm import tqdm 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    

    # define the model
    model_name = "modelname" #BioMistral-7.3B, Mistral0.2-7.3B, Mistral0.3-7.3B, LLaMa2-6.7B, LLaMa3.1-8B

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") 
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def _generate_response(model, inp):
        responses = []
        for i in range(1):
            out = model.generate(
            **inp,
            temperature=1e-12,
            do_sample=True,
            max_new_tokens=1)
            responses.append(tokenizer.decode(out[0], skip_special_tokens=True))
        return responses

    def collect_reasoning(csv_dir, tokenizer):
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        df_dataset = pd.DataFrame()
        inps = []
        reses = []

        path_file = "INPUT/PATH/NAME.csv"

        df = pd.read_csv(path_file)
        df['final_deid'] = df['final_deid'].astype(str)
        rm_strings = ['I, the teaching physician, have reviewed the images and agree with the report as written',
                    'I, the teaching physician, have reviewed the images, and agree  with the report as written',
                    "Critical results were communicated and documented using the Alert Notification  of Critical Radiology Results (ANCR) system",
                    "Critical results were communicated and documented using the Alert  Notification of Critical Radiology Results (ANCR) system"]

        # Remove each specified substring from each string in the column iteratively
        for string in rm_strings:
            df['final_deid'] = df['final_deid'].str.replace(string, '')
        
        # Define a function to remove "ATTESTATION: " and everything that comes after it
        def remove_substring(text, substring):
            index = text.find(substring)
            if index != -1:
                return text[:index]  # Return the text up to the found index
            else:
                return text 
        # Apply the function to the 'final_deid' column
        remove = "I, the teaching physician"
        df['final_deid'] = df['final_deid'].apply(lambda x: remove_substring(x, remove))

        remove = "ATTESTATION"
        df['final_deid'] = df['final_deid'].apply(lambda x: remove_substring(x, remove))

        remove = "Critical results were communicated"
        df['final_deid'] = df['final_deid'].apply(lambda x: remove_substring(x, remove))

        remove = "Electronically Signed by "
        df['final_deid'] = df['final_deid'].apply(lambda x: remove_substring(x, remove))
        conditions = ["any cancer", "progression/worsening", "response/improvement", 
                        "brain metastases", "bone/osseous metastases", "adrenal metastases", 
                        "liver/hepatic metastases", "lung/pulmonary metastases", 
                        "lymph node/nodal metastases", "peritoneal metastases"]

        def classify_impressions(df, column_name, conditions, prompt_text):
            
            # Reset the index of the DataFrame before running the loop. 
            # This will ensure that the DataFrame has a simple integer-based index which should align with loop's index variable.
            df = df.reset_index(drop=True)
            
            # Initialize a dictionary to store predictions
            predictions = {condition: [] for condition in conditions}
        
            processed_tokens = 0
            inps = []
            outs = []
            for index, impression in tqdm(df.iterrows(), desc="Starting"):
                # Constructing the full prompt with impression
                system_message = "You are a helpful assistant designed to analyze radiology reports."
                full_prompt = f"{prompt_text}\n\n{impression[column_name]}"
                if "model" in model_name:
                    prompt = f"[INST]system\n{system_message}[/INST]\n[INST]user\n{full_prompt}[/INST]\n[INST]assistant"
                else:
                    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant"
                prompt += "\n1."
                for i, condition in enumerate(conditions):
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=False
                    )

                    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                    generated_tokens = outputs.sequences[:,input_length:]
                    answer = tokenizer.decode(generated_tokens[0])
                    if "yes" in answer.lower():
                        label = "Yes"
                    elif "no" in answer.lower():
                        label = "No"
                    else:
                        if "model" in model_name:
                            yes_prob = np.max([outputs.scores[-1][:,3869].numpy()[0], outputs.scores[-1][:,4874].numpy()[0], outputs.scores[-1][:,22483].numpy()[0]])
                            no_prob = np.max([outputs.scores[-1][:,1939].numpy()[0], outputs.scores[-1][:,694].numpy()[0], outputs.scores[-1][:,11698].numpy()[0]])
                            if yes_prob > no_prob:
                                label = "Yes"
                            else: label = "No"
                        else:
                            yes_prob = np.max([outputs.scores[-1][:,5592].numpy()[0], outputs.scores[-1][:,5081].numpy()[0], outputs.scores[-1][:,2255].numpy()[0]])
                            no_prob = np.max([outputs.scores[-1][:,1770].numpy()[0], outputs.scores[-1][:,708].numpy()[0], outputs.scores[-1][:,7929].numpy()[0]])
                            if yes_prob > no_prob:
                                label = "Yes"
                            else: label = "No"
                    prompt += f" {label}\n{i}."
                    predictions[condition].append(label)
   
                # Save intermediate results every nth rows
                if (index + 1) % 500 == 0:
                    for condition, condition_predictions in predictions.items():
                        processed_predictions = condition_predictions[:index + 1]

                        # Get indices of the rows to update
                        indices_to_update = df.index[:index + 1]

                        # Use .loc to update the original DataFrame
                        df.loc[indices_to_update, f'{condition}_predicted'] = np.where(np.array(processed_predictions) == 'Yes', 1, 0)

                    # Save the updated part of the DataFrame
                    df.iloc[:index + 1].to_csv(f'{csv_path}/output_at_row_{index + 1}.csv', index=False)

                
            # Convert 'Yes'/'No' labels to binary (1/0) and add to DataFrame
            for condition, condition_predictions in predictions.items():
                df[f'{condition}_predicted'] = np.where(np.array(condition_predictions) == 'Yes', 1, 0)
            
            # df.to_csv(f'{csv_path}/final_output.csv', index=False)
            return df, predictions

        
        df_samples = pd.read_csv("PATH/SAMPLE/FOR/FEW_SHOT/NAME.csv")

        # prepare a dictionary for the samples
        dict_examples = {i+1: {'input': row['input'], 'output': row['output']} for i, row in df_samples.iterrows()}
        
        #identify the prompt
        prompt_text = "Identify if the following radiology impression text indicates (1) any cancer, (2) progression/worsening, (3) response/improvement, (4) brain metastases, (5) bone/osseous metastases, (6) adrenal metastases, (7) liver/hepatic metastases, (8) lung/pulmonary metastases, (9) lymph node/nodal metastases, (10) peritoneal metastases. Answer in Yes or No. Do not give an explanation."
        prompt_text += f"\n\nExample\nInput\n{dict_examples[1]['input']}\n\nOutput\n{dict_examples[1]['output']}\n\n"
        
        # generate the output
        dataframe, predoctions = classify_impressions(df, 'final_deid', conditions, prompt_text)
        dataframe.to_csv(f'{csv_path}/final_outs.csv', index=False)
    
    
    #PATH
    csv_path = "PATH/TO/SAVE/THE/OUTPUT"
    collect_reasoning(csv_path, tokenizer)


if __name__ == "__main__":
    main()