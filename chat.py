# Gemini Chatbot in Colab Enterprise with Simple Input UI

# Step 1: Install required packages
!pip install --upgrade google-cloud-aiplatform

# Step 2: Authenticate and configure Vertex AI
from google.colab import auth
from google.cloud import aiplatform

auth.authenticate_user()

PROJECT_ID = "your-project-id"  # Replace with your GCP project ID
REGION = "us-central1"          # Replace with your region (e.g., "us-west4")

aiplatform.init(project=PROJECT_ID, location=REGION)

# Step 3: Load Gemini model
from vertexai.preview.generative_models import GenerativeModel, ChatSession

model = GenerativeModel("gemini-1.5-pro")
chat_session = model.start_chat()

# Step 4: Define sample JSON data and system prompt
import json

json_data = {
    "products": [
        {"id": 1, "name": "Wireless Mouse", "price": 25.99, "stock": 14},
        {"id": 2, "name": "Keyboard", "price": 45.50, "stock": 7},
        {"id": 3, "name": "USB-C Hub", "price": 19.99, "stock": 0}
    ],
    "orders": {
        "ORD123": {"status": "shipped", "items": [1, 2]},
        "ORD124": {"status": "processing", "items": [3]}
    }
}

system_prompt = """
You are a virtual assistant for an online electronics store.
You answer questions about product availability, prices, and order statuses.
Always be concise and helpful. If data is missing, say so.
"""

# Step 5: Define chatbot query function
def query_chatbot(user_input):
    formatted_data = json.dumps(json_data, indent=2)
    prompt = f"""
{system_prompt}

Here is the current product and order data:
{formatted_data}

User question: {user_input}
"""
    response = chat_session.send_message(prompt)
    return response.text

# Step 6: Simple input UI using IPython widgets
import ipywidgets as widgets
from IPython.display import display

input_box = widgets.Text(
    value='',
    placeholder='e.g., What is the status of order ORD123?',
    description='Ask:',
    disabled=False,
    layout=widgets.Layout(width='100%')
)

output_area = widgets.Output()

def handle_input(sender):
    with output_area:
        output_area.clear_output()
        response = query_chatbot(input_box.value)
        print("Bot:", response)

input_box.on_submit(handle_input)

print("\n🤖 Chatbot Ready! Ask your question below:")
display(input_box, output_area)
