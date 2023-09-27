import logging

import openai
from flask import Flask, render_template, request, jsonify
import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
import requests
from langchain.vectorstores import Chroma

app = Flask(__name__, static_url_path='/static')
"""
Set your API key first and keep them SAFE!!
"""
# os.environ["OPENAI_API_KEY"] ="put your key here"
# openai.api_key = os.environ["OPENAI_API_KEY"]
# os.environ['GOOGLE_API_KEY'] = 'put your key here'
# os.environ['GOOGLE_CSE_ID'] = 'put your key here'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'
# ALPHA_VANTAGE_API_KEY = 'put your key here'
# NEWS_API_KEY = 'put your key here'

# # Configure logging
# log_file = 'app.log'
# logging.basicConfig(filename=log_file, level=logging.DEBUG)
# logging.info('start')
stock_data = {
    'aapl': {'Price': 186, '1-year lowest': 124, '1-year highest': 187, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'msft': {'Price': 335, '1-year lowest': 213, '1-year highest': 351, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'goog': {'Price': 123, '1-year lowest': 83, '1-year highest': 129, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'amzn': {'Price': 129, '1-year lowest': 81, '1-year highest': 146, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'nvda': {'Price': 422, '1-year lowest': 108, '1-year highest': 440, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'tsla': {'Price': 256, '1-year lowest': 102, '1-year highest': 314, 'GPT reasoning and ranking': 8,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Buy'},
    'brkb': {'Price': 335, '1-year lowest': 260, '1-year highest': 341, 'GPT reasoning and ranking': 9,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'OutPerform'},
    'meta': {'Price': 288, '1-year lowest': 88, '1-year highest': 290, 'GPT reasoning and ranking': 8,
             'Rating of next 1-year': 'UnderPerform', 'Rating of next 10-year': 'Hold'}
}
search = GoogleSearchAPIWrapper()
# run the redis server first
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)
loader = TextLoader("./data/data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="qury from data.txt",
        func=qa.run,
        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or "
                    "fully formed question.",
    )
]
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to 
the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


@app.route("/", methods=["GET", "POST"])
def home():
    error_message = None
    value = None
    news = None
    balance_sheet = None
    recommendations = None
    if request.method == "POST":
        try:
            symbol = request.form["symbol"]
            symbol_lowercase = symbol.lower()
            if symbol_lowercase in stock_data:
                recommendations = stock_data[symbol_lowercase]
            # Fetch company news from News API
            news_url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}'
            response = requests.get(news_url)
            if response.status_code == 200:
                news_data = response.json()
                articles = news_data.get('articles')
                if articles:
                    news = [article.get('title') for article in articles][:5]  # Get top 5 news titles
                    news = list(set(news))
                # Fetch balance sheet from Alpha Vantage API
                value = {}

                # Company Overview
                overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                overview_response = requests.get(overview_url)
                overview_data = overview_response.json()

                value['company_name'] = overview_data.get('Name')
                value['sector'] = overview_data.get('Sector')
                value['company_description'] = overview_data.get('Description')

                # Global Quote
                quote_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                quote_response = requests.get(quote_url)
                print(quote_response.json())
                quote_data = quote_response.json()['Global Quote']

                value['current_price'] = round(float(quote_data.get('05. price')), 2)
                value['previous_close'] = round(float(quote_data.get('08. previous close')), 2)
                value['open_price'] = round(float(quote_data.get('02. open')), 2)

                # Balance Sheet
                balance_sheet_url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                balance_sheet_response = requests.get(balance_sheet_url)
                balance_sheet_data = balance_sheet_response.json()
                balance_sheet = balance_sheet_data.get('quarterlyReports')  # or 'annualReports' for annual data

                # INCOME_STATEMENT for earnings data
                income_statement_url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                response = requests.get(income_statement_url)
                if response.status_code == 200:
                    data = response.json()
                    income_statement = data.get('annualReports')[0]  # Get the most recent annual data

                    if 'netIncome' in income_statement and 'SharesOutstanding' in overview_data:
                        net_income = float(income_statement['netIncome'])
                        shares_outstanding = float(overview_data['SharesOutstanding'])
                        eps = net_income / shares_outstanding
                        value['pe_ratio'] = round(float(value['current_price']) / eps, 2)
                    else:
                        value['pe_ratio'] = 'N/A'

                    if 'totalRevenue' in income_statement:
                        total_sales = float(income_statement['totalRevenue'])
                        value['ps_ratio'] = round((float(value['current_price']) * shares_outstanding) / total_sales, 2)
                    else:
                        value['ps_ratio'] = 'N/A'

                    if 'totalShareholderEquity' in balance_sheet[0]:
                        value['roe'] = round(net_income / float(balance_sheet[0][
                                                                    'totalShareholderEquity']),
                                             2)  # Take the first item of the quarterlyReports which is the most recent data
                    else:
                        value['roe'] = 'N/A'

                else:
                    error_message = "An error occurred when fetching income statement data. Please try again later."

                # Earnings
                earnings_url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                earnings_response = requests.get(earnings_url)
                if earnings_response.status_code == 200:
                    earnings_data = earnings_response.json()
                    annual_earnings = earnings_data.get('annualEarnings')
                    quarterly_earnings = earnings_data.get('quarterlyEarnings')

                print(recommendations)
        except Exception as e:
            error_message = "An error occurred. Please try again later."
            logging.error(str(e), exc_info=True)

    logging.info(news)
    logging.info(balance_sheet)
    return render_template("home.html", value=value, news=news, balance_sheet=balance_sheet,
                           error_message=error_message, recommendations=recommendations)


@app.route("/chat", methods=["POST"])
def chat():
    # logging.info("Chat API called")
    print("Chat API called")
    loader = TextLoader("./data/data.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])
    if request.method == "POST":
        try:
            data = request.get_json()
            user_message = data["message"]
            logging.info(user_message)
            # Call the OpenAI Chat API
            # completion = openai.ChatCompletion.create(
            #     model='gpt-3.5-turbo',
            #     messages=[
            #         {'role': 'user', 'content': user_message}
            #     ],
            #     temperature=0
            # )
            llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=memory
            )
            # data_response = index.query(user_message,llm=ChatOpenAI(),memory=(ConversationBufferMemory(llm=ChatOpenAI())))
            google_response = agent_chain.run(user_message)

            # logging.info(chat_response)
            # print("data"+data_response)
            logging.info("response:" + google_response)
            return jsonify({"response": google_response})
        except Exception as e:
            logging.error(str(e), exc_info=True)
            return jsonify({"error": "An error occurred. Please try again later."})

    return jsonify({"error": "Invalid request"})


def decide_best_response(user_message, google_results, index_results):
    # Feed the results to the GPT model and have it decide which one is more relevant
    # You would need to define how GPT makes this decision

    # Initialize tools

    # Define prefix, suffix, and input_variables
    prefix = """answering the user questions as best you can. You have to choose one of these two answer:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    # Create the prompt
    prompt = ZeroShotAgent.create_prompt(
        prefix=prefix,
        suffix=suffix,
    )

    # Run the LLMChain
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

    # Prepare the model input
    model_input = {
        "input": user_message,
        "chat_history": google_results,
        "agent_scratchpad": index_results
    }

    # Run the model with the prepared input
    model_output = llm_chain.run(model_input)

    return model_output


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import subprocess
import uuid


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    print("transcribe_audio called")

    unique_id = str(uuid.uuid4())
    filename = f'uploads/audio_{unique_id}.mp3'
    convert = f'uploads/audio_converted_{unique_id}.mp3'

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files['audio']
    audio_file.save(filename)
    audio_file.close()

    # Convert the saved audio file to mp3 format using ffmpeg
    output_filename = os.path.splitext(filename)[0] + f"_converted_{unique_id}.mp3"
    print(output_filename)

    subprocess.run(['ffmpeg', '-i', filename, output_filename], capture_output=True, text=True)

    # Print ffmpeg output for debugging

    # Check if conversion was successful
    if not os.path.exists(output_filename):
        return jsonify({"error": "File conversion failed"}), 500

    try:
        transcribed_text = transcribe_file_to_text(output_filename)
        print(transcribed_text)
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    return jsonify({"text": transcribed_text})


def transcribe_file_to_text(file_path):
    # Assuming you're using OpenAI's Whisper
    with open(file_path, 'rb') as audio_file:
        result = openai.Audio.translate("whisper-1", audio_file)
    return result["text"]


import pyaudio
import wave
import openai


def record_and_transcribe():
    duration = 5  # seconds
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=44100,
                           input=True,
                           frames_per_buffer=1024)

    audio_data = []
    print(f"Recording started for {duration} seconds...")

    for _ in range(0, int(44100 / 1024 * duration)):
        data = audio_stream.read(1024)
        audio_data.append(data)

    audio_stream.close()
    print(f"Recording completed. Captured {len(audio_data)} chunks of audio.")

    # Write audio_data to a WAV file
    output_filename = "temp_recording_fixed_duration.wav"
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(audio_data))

    print(f"Saved recording to {output_filename}.")

    # Transcribe using OpenAI's Whisper
    with open(output_filename, 'rb') as audio_file:
        result = openai.Audio.translate("whisper-1", audio_file)

    return result["text"]


if __name__ == '__main__':
    # print(record_and_transcribe())
    # run_quickstart()
    app.run(host='0.0.0.0', port=8080)
