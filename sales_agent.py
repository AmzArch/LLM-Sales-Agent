import os
import re
import json

from dotenv import load_dotenv

load_dotenv()

from typing import Any, Callable, Dict, List, Union
from pydantic import BaseModel, Field
import dspy
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chromadb
from chromadb.utils import embedding_functions
from dspy.retrieve.chromadb_rm import ChromadbRM
import json


llm = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens = 1000)
dspy.settings.configure(lm=llm)

product_catalog = "sales_agent/sample_product_catalog.txt"

# Set up a knowledge base
def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product knowledge base is simply a text file.
    """
    # load product catalog
    with open(product_catalog, "r") as f:
        product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)

    llm = ChatOpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base

knowledge_base = setup_knowledge_base("sales_agent/sample_product_catalog.txt")

product_price_id_map = {
    "ai-consulting-services": "price_1Ow8ofB795AYY8p1goWGZi6m",
    "Luxury Cloud-Comfort Memory Foam Mattress": "price_1Owv99B795AYY8p1mjtbKyxP",
    "Classic Harmony Spring Mattress": "price_1Owv9qB795AYY8p1tPcxCM6T",
    "EcoGreen Hybrid Latex Mattress": "price_1OwvLDB795AYY8p1YBAMBcbi",
    "Plush Serenity Bamboo Mattress": "price_1OwvMQB795AYY8p1hJN2uS3S",
}
with open("example_product_price_id_mapping.json", "w") as f:
    json.dump(product_price_id_map, f)

def product_price_id_mapping(product_price_id_mapping_path):
    # Load product_price_id_mapping from a JSON file
    with open(product_price_id_mapping_path, "r") as f:
        product_price_id_mapping = json.load(f)

    # Serialize the product_price_id_mapping to a JSON string for inclusion in the prompt
    product_price_id_mapping_json_str = json.dumps(product_price_id_mapping)

    # Dynamically create the enum list from product_price_id_mapping keys
    enum_list = list(product_price_id_mapping.values()) + [
        "No relevant product id found"
    ]
    enum_list_str = json.dumps(enum_list)

    return product_price_id_mapping_json_str, enum_list_str


class DecideWhichAgent(dspy.Signature):
    """
    You are a routing agent which routes customers to either a sales agent or a customer account management agent.
    Given the following conversation history, decide wether a sales agent or a customer account management agent would be better suited to solving the user's latest query.
    If the conversation history is empty or an empty string and there is not enough information to decide, assume it is a sales related query and just selected the sales agent.
    Answer only with 1 for sales agent and 2 for customer support
    """
    
    conversation_history = dspy.InputField(desc="The entire conversation histiry between a customer and a sales agent till now")
    answer = dspy.OutputField(desc="A number 1 or 2 or 0")

decideAgent = dspy.Predict(DecideWhichAgent)

class StageAnalyzer(dspy.Signature):
    """
    Use the conversation history to decide which stage the conversation is at.
    Now determine what should be the next immediate conversation stage for the agent in the sales conversation by selecting ony from the following options:
    1. Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.
    2. Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
    3. Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
    4. Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
    5. Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
    6. Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
    7. Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.

    Only answer with a number between 1 through 7 with a best guess of what stage should the conversation continue with. 
    The answer needs to be one number only, no words.
    If there is no conversation history, output 1.
    Do not answer anything else nor add anything to you answer.
    """

    conversation_history = dspy.InputField(desc="The entire conversation histiry between a customer and a sales agent till now")
    answer = dspy.OutputField()

StageAnalysis = dspy.Predict(StageAnalyzer)

class KnowledgeAugmentation(dspy.Signature):
    """
    Use the conversation history to decide whether we need to retrieve context from the product catalog to retrieve knowledge.
    Reply with a 0 if we do not need to retrieve knowledge or with a query to run of what is needed from the product catalog
    """

    conversation_history = dspy.InputField(desc="The entire conversation history between a customer and a sales agent till now")
    answer = dspy.OutputField(desc="0 or a query")

KnowledgeAugmentationQuery = dspy.Predict(KnowledgeAugmentation)

class SalesConversationReply(dspy.Signature):
    """
    You are a sales agent. The following are the details of your role and company.
    If you're asked about where you got the user's contact information, say that you got it from public records.
    Keep your responses in short length to retain the user's attention. Never produce lists, just answers.
    You must respond according to the previous conversation history and the stage of the conversation you are at.
    Only generate one response at a time! When you are done generating, end with '<END_OF_TURN>' to give the user a chance to respond.
    Example:
        Conversation history: 
        {salesperson_name}: Hey, how are you? This is {salesperson_name} calling from {company_name}. Do you have a minute? <END_OF_TURN>
        User: I am well, and yes, why are you calling? <END_OF_TURN>
        {salesperson_name}:
        End of example.
         
    """

    salesperson_name = dspy.InputField(desc="The name you are operating under")
    salesperson_role = dspy.InputField(desc="This is the role you are replying to the customer as")
    company_name = dspy.InputField(desc="You work at this company")
    company_business = dspy.InputField(desc="The company does this business")
    company_values = dspy.InputField(desc="These are the company values")
    conversation_purpose = dspy.InputField(desc="The reason you are contacting a potential customer")
    conversation_type = dspy.InputField(desc="Your means of contacting the prospect")
    conversation_stage = dspy.InputField(desc="Current conversation stage")
    conversation_history = dspy.InputField(desc="the Conversation History")
    context = dspy.InputField(desc = "Any context retrieved to answer the query")
    answer = dspy.OutputField()

SalesReply = dspy.Predict(SalesConversationReply)

conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
    "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
    "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
    "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
    "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
}

class getProductIDfromQuery(dspy.Signature):
    """
    You are an expert data scientist and you are working on a project to recommend products to customers based on their needs.
    Given a query and the following product id mapping return the price id that is most relevant to the query.
    ONLY return the price id, no other text. If no relevant price id is found, return 'No relevant price id found'.
    Your output will follow this schema while replacing enum_list_str from the inputs given:
    {{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Price ID Response",
    "type": "object",
    "properties": {{
        "price_id": {{
        "type": "string",
        "enum": {enum_list_str}
        }}
    }},
    "required": ["price_id"]
    }}
    Return a valid directly parsable json, dont return in it within a code snippet or add any kind of explanation!!
    """
    query = dspy.InputField(desc="User Query")
    product_price_id_mapping_json_str = dspy.InputField(desc="Product ID Mapping")
    enum_list_str = dspy.InputField(desc = "The enum_list_str to be filled out in the json output")
    answer = dspy.OutputField()

product_id_from_Query = dspy.Predict(getProductIDfromQuery)

def generate_stripe_payment_link(query: str) -> str:
    """Generate a stripe payment link for a customer based on a single query string."""

    PRODUCT_PRICE_MAPPING = "example_product_price_id_mapping.json"
    product_price_id_mapping_json_str, enum_list_str = product_price_id_mapping(product_price_id_mapping_path=PRODUCT_PRICE_MAPPING)

    # use LLM to get the price_id from query
    product_id = product_id_from_Query(query = query, product_price_id_mapping_json_str=product_price_id_mapping_json_str, enum_list_str=enum_list_str)
    price_id = product_id.answer.strip()
    price_id = json.loads(price_id)
    payload = json.dumps(
        {"prompt": query, **price_id}
    )
    return payload

config = dict(
    salesperson_name="Ted Lasso",
    salesperson_role="Business Development Representative",
    company_name="Sleep Haven",
    company_business="Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer free delivery on all matresses.",
    company_values="Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service.",
    conversation_purpose="find out whether they are looking to achieve better sleep via buying a premier mattress.",
    conversation_history=None,
    conversation_type="call",
    conversation_stage=conversation_stages.get("1"),
    context = "None"
)

class ResponseEngine():
    def __init__(self, config) -> None:
        self.conversation_history = config["conversation_history"]
        self.salesperson_name=config["salesperson_name"]
        self.salesperson_role=config["salesperson_role"]
        self.company_name=config["company_name"]
        self.company_business=config["company_business"]
        self.company_values=config["company_values"]
        self.conversation_purpose=config["conversation_purpose"]
        self.conversation_type=config["conversation_type"]
        self.conversation_stage=config["conversation_stage"]
        self.agent = None

        if self.conversation_history =="" or self.conversation_history==None or self.conversation_history==[]:
            self.conversation_stage = config.get("1")
            self.conversation_history = None

    def reply(self):
        agent = decideAgent(conversation_history = self.conversation_history).answer

        if agent == "1":
            query = KnowledgeAugmentationQuery(conversation_history = self.conversation_history).answer
            print(query)
            if query == '0':
                stage = StageAnalysis(conversation_history = self.conversation_history)
                reply = SalesReply(**(config | {"conversation_stage": conversation_stages.get(stage)})).answer
            else:
                context = knowledge_base.run(query)
                stage = StageAnalysis(conversation_history = self.conversation_history)
                reply = SalesReply(**(config | {"conversation_stage": conversation_stages.get(stage), "conext": context})).answer

        else:
            reply = "Doing Customer Account Management API Calls"
        if self.conversation_history == None:
            self.conversation_history = ""
        self.conversation_history = self.conversation_history + "\n" + self.salesperson_name + ": " + reply
        config["conversation_history"] = self.conversation_history
        
        return reply, self.conversation_history
    

response_engine = ResponseEngine(config=config)

# response_engine.reply()

# Function to handle the conversation loop
def conversation_loop():
    while True:
        # User input
        user_reply = input("You: ")
        if user_reply.lower() == 'exit':
            print("Conversation ended.")
            break
        
        # Update conversation history
        config["conversation_history"] += f"\nUser: {user_reply}"
        
        # Get response from the response engine
        system_response = response_engine.reply()
        
        # Display the system response
        print(system_response)

response_engine.reply()
# # Start the conversation loop
# if __name__ == "__main__":
#     conversation_loop()