# Standard library imports
import csv
import logging
import json
import time

from functools import wraps
from itertools import islice
from typing import List, Dict, Optional, Any, Iterator

# Third-party imports
import openai
import pandas as pd

# Internal imports
from utils.utils import save_to_csv
from utils.decorators import retry_request

# Configure logging
logging.basicConfig(level=logging.INFO)

TRIPLET_TYPES = {
    "SVO": "Subject-Verb-Object",
    "EAT": "Entity-Action-Time",
    "PBL": "Person-BornIn-Location",
    "OFY": "Organization-Founded-Year",
    "PMM": "Product-MadeFrom-Material",
    "EOL": "Event-OccursIn-Location",
    "EIT": "Entity-IsA-Type",
    "AWA": "Article-WrittenBy-Author",
    "PUT": "Project-Uses-Technology",
    "PAI": "Person-Attended-Institution",
    "OWP": "Object-WasProduced-Date",
    # "SHE": "Software-Has-Error",
    "CSS": "Company-Sells-Service",
    # "DIE": "Document-Is-Encrypted",
    "RLE": "Research-LeadBy-Expert",
    # "DCH": "Device-Contains-Hardware",
    # "AFG": "Animal-FoundIn-Geolocation",
    "ROR": "Report-Offers-Recommendation",
    # "GBA": "Game-BelongsTo-AppStore",
    "PCI": "Person-CollaboratesWith-Institution",
    "ERE": "Entity-Relationship-Entity",
    "EAV": "Entity-Attribute-Value",
    "EAO": "Entity-Action-Outcome",
    "EPE": "Entity-PartOf-Entity",
    "TEE": "Time-Event-Entity" 
}

class DataSyntheticGenerator:
    """A class for generating synthetic data using the OpenAI API."""
    
    def __init__(self) -> None:
        pass

    @retry_request(wait_seconds=2, max_retries=3)
    def _make_request(self, prompt: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Makes a request to the OpenAI API and returns the response.
        
        Parameters:
        - prompt (List[Dict[str, str]]): The prompt to send to the API.

        Returns:
        - Dict[str, Any]: The API response.
        """
        
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=prompt
        )
    
    def get_sample_text(self, theme: str, triplets_with_description: str) -> Optional[Dict[str, Any]]:
        """Generates a sample text prompt based on the description and fetches the result.
        
        Parameters:
        - description (str): A description of the text to generate.

        Returns:
        - Dict[str, Any]: The API response.
        """
        system_message = f"You are an expert in circular economy. You always answer in spanish."
        user_message = f"Please provide 10 different long texts for the theme {theme} in the context of circular economy, make sure each of these paragraphs include enough material to retrieve at least one of each of the following triplets: {triplets_with_description}. Respond with plain text in spanish separating each text with a line break."
        user_message = f"""
            Im working in triplet extraction of type {triplets_with_description} and I want to generate a dataset.
            Generate five long paragraphs in Spanish about circular economy for {theme} theme.
            All paragraphs must contain enough information to extract the triplets described above but without mentioning anything about the type of triplet .
            Respond with plain text in spanish separating each paragraph with a line break.
        """
        prompt = [
            # {"role": "system", "content": system_message}, 
            {"role": "user", "content": user_message}
        ]
        
        print(prompt)
        response =  self._make_request(prompt)
        print(response)
        return response

    def generate_triplets(self, triplet_type: str, text: str) -> Optional[Dict[str, Any]]:
        """Generates triplets and writes them to a CSV file.
        
        Parameters:
        - triplet_type (str): The type of triplet to generate.
        """
        system_message = f"You are a helpful tool that extracts triplets from spanish text."
    
        user_message = f"""
            Please extract relations of type {triplet_type} from the given Spanish text: {text}.
            Output the extracted relations as plain text in a CSV format, without a header.
            Each triplet must be separated by a line break and each component of the triplet must be enclosed between double quotes and separated by a ";" and at the end include the triplet type enclosed by double quotes.
            Here is an example: "component1";"relation";"component2";"tryplet type"
        """

        # user_message = f"""
        #     Please extract specific types of relations from the provided spanish text: {text} based on the designated triplet types. The triplet types can be {triplet_type}.
        #     Output the extracted relations in plain text, formatted as a CSV in spanish without headers. Each entry should adhere to the following format: "component1";"relation";"component2";"triplet_type". Separate each entry by a line break.

        #     For example, a sample output entry might look like:
        #     "María";"come";"pera";"SVO"
        # """
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return self._make_request(prompt)

    def generate_text_and_triplets(self, triplet_type: str) -> Optional[Dict[str, Any]]:
        """Generates triplets and writes them to a CSV file.
        
        Parameters:
        - triplet_type (str): The type of triplet to generate.
        """
        system_message = f"You are an assistant expert in circular economy and construction materials that generates text in spanish and extracts triplets."
    
        user_message = f"""
            The Goal is to generate samples of text in spanish about construction or circular economy and the relations extracted from it. 
            I need you to generate text in spanish of different kind (like a user manual, a website, an oficial document, a legal document, etc...) of at least 100 words each
            and the text must contain enough information to extract at least three types of triplet of the ones mentioned. The types of triplet are: {triplet_type}.
            The output should be in json format for example [{{"text": "generated text here", triplets: [{{"type_of_triplet":"type of triplet", "component1":"first component of the triplet", "component2": "second component of the triplet", "component3":"third component of the triplet"}}]}}].
            Answer only the json as plaintext and everything in Spanish with at least 10 entries, do not include anything else in the response.
        """

        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return self._make_request(prompt)

def generate_paragraphs_old(generator: DataSyntheticGenerator):
    
    # Load themes
    themes = pd.read_csv("data/themes.csv", sep=";")
    # Convert the dictionary items to a list and then slice the list in half
    dict_items = list(TRIPLET_TYPES.items())
    mid_idx = len(dict_items) // 2

    # Create two new dictionaries from the slices
    first_half_dict = dict(dict_items[:mid_idx])
    second_half_dict = dict(dict_items[mid_idx:])

    # Create the string
    first_half_concatenated_string = ", ".join([f"{key} ({value})" for key, value in first_half_dict.items()])
    second_half_concatenated_string = ", ".join([f"{key} ({value})" for key, value in second_half_dict.items()])

    only_keys_first_half_concatenated_string = ", ".join([f"{key}" for key in first_half_dict.items()])
    only_keys_second_half_concatenated_string = ", ".join([f"{key}" for key in second_half_dict.items()])

    paragraph_csv = "paragraph;triplet_type\n"
    for index, row in themes.iterrows():
        theme = row["theme"]
        for i in range(10):
            logging.info(f"requesting paragraphs for theme {theme} iteration {i}.")
            response = generator.get_sample_text(theme=row["theme"], triplets_with_description=first_half_concatenated_string)['choices'][0]['message']['content']
            response = "\n".join([f"{line};{only_keys_first_half_concatenated_string}" for line in response.split("\n")])
            paragraph_csv+= response + "\n"

            response = generator.get_sample_text(theme=row["theme"], triplets_with_description=second_half_concatenated_string)['choices'][0]['message']['content']
            response = "\n".join([f"{line};{only_keys_second_half_concatenated_string}" for line in response.split("\n")])
            paragraph_csv+= response + "\n"
            
            save_to_csv(filename="data/paragraphs.csv", content=paragraph_csv, append=False)

def generate_paragraphs(generator: DataSyntheticGenerator):
    
    # Load themes
    themes = pd.read_csv("data/themes.csv", sep=";")
    # Convert the dictionary items to a list and then slice the list in half
    dict_items = list(TRIPLET_TYPES.items())
    paragraph_csv = "paragraph;triplet_type\n"

    for i in range(0, len(dict_items), 3):
        chunk_dict = dict(dict_items[i:i + 3])

        # Create concatenated strings
        concatenated_string = ", ".join([f"{key} ({value})" for key, value in chunk_dict.items()])
        only_keys_concatenated_string = ", ".join([f"{key}" for key, value in chunk_dict.items()])
        
        for index, row in themes.iterrows():
            theme = row["theme"]
            for j in range(25):
                logging.info(f"Requesting paragraphs for theme {theme}, iteration {j}.")
                
                response = generator.get_sample_text(theme=row["theme"], triplets_with_description=concatenated_string)['choices'][0]['message']['content']
                response = "\n".join([f"{line};{only_keys_concatenated_string}" for line in response.split("\n")])
                
                paragraph_csv += response + "\n"

                # Save to CSV
                save_to_csv(filename="data/paragraphs.csv", content=paragraph_csv, append=False)


def generate_triplets(generator: DataSyntheticGenerator):
    # Load themes
    paragraphs = pd.read_csv("data/paragraphs.csv", sep=";")
    paragraph_csv = "id;text;component1;relation;component2;triplet_type\n"

    for index, row in paragraphs.iterrows():
        if index < 1383:
            continue
        paragraph = row["paragraph"]
        triplet_types = row["triplet_type"]
        logging.info(f"Evaluating text: {paragraph}")
        triplet_with_desc = ', '.join([f"{t.strip()} ({TRIPLET_TYPES[t.strip()]})" for t in triplet_types.split(", ")])

        logging.info(f"triplet type: {triplet_with_desc}")
        response = generator.generate_triplets(triplet_type=triplet_with_desc,
                                                text=paragraph)
        response = response['choices'][0]['message']['content'].split("\n")

        for r in response:
            text = f"""
                "{index}";"{paragraph.replace('"','')}";"{r}"
            """
            print(text)
            paragraph_csv += text

            save_to_csv(filename="data/triplet_paragraph.csv", content=paragraph_csv, append=False)

def chunks(data: Dict[Any, Any], chunk_size: int) -> Iterator[Dict[Any, Any]]:
    """
    Yields successive n-sized chunks from a dictionary.

    :param data: The dictionary to split.
    :param chunk_size: The size of each chunk.
    :return: An iterator that yields dictionary chunks.
    """
    logging.info("Starting to create chunks from dictionary.")
    it = iter(data.items())
    while True:
        chunk = dict(islice(it, chunk_size))
        if not chunk:
            logging.info("No more chunks to process. Exiting.")
            return
        logging.debug(f"Generated chunk: {chunk}")
        yield chunk

def generate_triplets_json(generator: DataSyntheticGenerator) -> None:
    paragraph_csv = "id;text;component1;relation;component2;triplet_type\n"
    last_index = 1383

    for i in range(2000):
        logging.info(f"Iteration {i}")

        for index, chunk in enumerate(chunks(TRIPLET_TYPES, 3)):
            triplet_with_desc = ', '.join([f"{key} ({value})" for key, value in chunk.items()])
            logging.info(f"requestion triplets: {triplet_with_desc}")
            try:
                response = generator.generate_text_and_triplets(triplet_type=triplet_with_desc)
                response = response['choices'][0]['message']['content']
                response_data = json.loads(response)
                logging.info(response_data)
                for entry in response_data:
                    text = entry["text"]
                    triplets = entry["triplets"]
                    
                    for triplet in triplets:
                        triplet_type = triplet["type_of_triplet"]
                        component1 = triplet["component1"]
                        component2 = triplet["component2"]
                        component3 = triplet["component3"]
                        
                        # Create CSV line and append it to paragraph_csv
                        csv_line = f"{i+last_index};{text};{component1};{component2};{component3};{triplet_type}\n"
                        paragraph_csv += csv_line
                    save_to_csv(filename="data/triplet_paragraph.csv", content=paragraph_csv, append=False)

            except Exception as e:
                logging.error(f"An error occurred: {e}")



if __name__ == "__main__":
    generator = DataSyntheticGenerator()

    # generate_paragraphs(generator)
    # generate_triplets(generator=generator)
    generate_triplets_json(generator=generator)
    # for triplet_type, description in TRIPLET_TYPES.items():
        
        # for _ in range(100):  # Adjust this for 10K samples
            # generator.generate_triplets(triplet_type=triplet_type, description=description)
