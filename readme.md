# Chatbot using FLAN-T5-XL and Streamlit

## Project Objective

The objective of this project is to create a chatbot using the FLAN-T5-XL model from Hugging Face's Transformers library. The chatbot is capable of having a conversation with users based on a predefined set of initial prompts. The chatbot uses Streamlit for the user interface, allowing users to interact with the bot in a web browser.

## Technologies Used

- **Python**: The backend logic of the chatbot is implemented using Python programming language.
- **PyTorch**: PyTorch is used as the deep learning framework to work with the FLAN-T5-XL model.
- **Transformers**: The Transformers library by Hugging Face is used to load the pretrained FLAN-T5-XL model and the necessary tokenizer.
- **Streamlit**: Streamlit is used to create the web interface for the chatbot.
- **CUDA**: CUDA environment variables are set to ensure compatibility with NVIDIA's CUDA platform (if available).

## Setup and Installation

1. Ensure you have Python 3.8 or later installed.
2. Clone this repository to your local machine.

3. Create a virtual environment in the project directory:
   ```
   python -m venv venv
   ```

Activate the virtual environment:
On Windows:
```
.\venv\Scripts\activate```

On Linux/Mac:

```source venv/bin/activate```

Install the necessary Python packages:


``` pip install -r requirements.txt ```

Run the Streamlit app:
``` streamlit run app.py ```


Open your web browser and go to http://localhost:8501.  
You will see the chatbot interface with a text input field and a "Send" button.  
Type your message in the text input field and click "Send" to get a response from the chatbot.  
The chat history will be displayed in a text area below the chat input field.  
Contributing  
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.  

License  
This project is open-source and available under the MIT License.  

Acknowledgements
The pretrained FLAN-T5-XL model is provided by Hugging Face.  
The initial version of this chatbot was created as a part of a collaborative project.  