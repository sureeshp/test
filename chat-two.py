# Vertex AI Gemini Chatbot in Colab Enterprise

# Step 1: Install Vertex AI SDK
!pip install --upgrade google-cloud-aiplatform

# Step 2: Authenticate and configure Vertex AI
from google.colab import auth
from google.cloud import aiplatform

auth.authenticate_user()

PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project ID
REGION = "us-central1"              # Replace with your region

aiplatform.init(project=PROJECT_ID, location=REGION)

# Step 3: Load Gemini model via Vertex AI
from vertexai.preview.generative_models import GenerativeModel, ChatSession

model = GenerativeModel("gemini-1.5-pro")
chat_session = model.start_chat()

# Step 4: Define system prompt and query function
system_prompt = """
You are a helpful assistant for a tech support chatbot.
Answer questions clearly and concisely based on user input.
"""

def query_gemini(user_input):
    prompt = f"""
{system_prompt}

User: {user_input}
Assistant:"""
    response = chat_session.send_message(prompt)
    return response.text

# Step 5: Simple chatbot interface using IPython widgets
import ipywidgets as widgets
from IPython.display import display

input_box = widgets.Text(
    placeholder='e.g., How do I reset my password?',
    description='Ask:',
    layout=widgets.Layout(width='100%')
)

output_area = widgets.Output()

def handle_submit(sender):
    with output_area:
        output_area.clear_output()
        response = query_gemini(input_box.value)
        print("Bot:", response)

input_box.on_submit(handle_submit)

print("\n🤖 Gemini Chatbot Ready! Ask your question below:")
display(input_box, output_area)
