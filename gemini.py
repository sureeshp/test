
import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import re
# ── Load API key and environment ─────────────────────────────────────
load_dotenv()




# Configure the API key
# It's recommended to set GOOGLE_API_KEY as an environment variable
# For demonstration, you could directly assign:
# genai.configure(api_key="YOUR_GOOGLE_API_KEY")

from groq import Groq

#client=ChatGroq(model=os.environ.get("LITELLM_MODEL"))

#OPENAI_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_KEY:
#    st.error("❌ OPENAI_API_KEY not found in your .env file.")
#    st.stop()
client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# ── Load data files ──────────────────────────────────────────────────
with open("data/products.json", "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)
with open("data/premium.json", "r", encoding="utf-8") as f:
    PREMIUM = json.load(f)
with open("data/p_description.json", "r", encoding="utf-8") as f:
    P_DESCRIPTIONS = json.load(f)
with open("data/transaction.json", "r", encoding="utf-8") as f:
    TR = json.load(f)

# ── Build product context ────────────────────────────────────────────
def build_product_context(products):
    return "\n".join(
        f"- {p['name']} ({p['deviceType']}, Rental: {p['rentalCostPerMonth']}, "
        f"Printer: {p['builtInPrinter']}, Connectivity: {p['connectivity']}, "
        f"Battery: {p['batteryLife']}, Contract: {p['contractRequired']}, "
        f"Ideal For: {p['idealFor']})"
        for p in products
    )

PRODUCT_CONTEXT = build_product_context(PRODUCTS)

# ── System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are a warm, professional AI assistant for a Bank, specializing exclusively in recommending Cardnet POS devices. You must only retrieve and use data from the provided files: PRODUCT_CONTEXT, P_DESCRIPTIONS, TR and PREMIUM. Ask only the clarifying questions you need—no intros or chit-chat. Avoid emojis and unnecessary symbols.
My files are {PRODUCT_CONTEXT,PREMIUM,P_DESCRIPTIONS,TR}
# *****Always use the data given in the json files and don't create any other benifits/features on your own******
# *****uask most relevent 3 question based on data and if you still not able to you suggest products ask more,if we have given more than one products ask for few other questions to get more filtering of products******
# Always use the {TR} file for the  transaction fees and display the variable and fixed fee together and give that as output for transaction fees,dont display Fixed +varaible transaction fees for calculations also and display the output.

Clarifying Questions Logic  
- If the user requests transaction fees under a threshold, immediately ask:  
  • “How many card transactions per month?”  
  • “What is the average value of each card transaction?”  
  Once you have both, in that same first response include:  
  Estimated monthly transaction fee: £X.XX  
  (Provide the fee breakdown formula only if the user asks a follow-up.)
 
- For all other cases, ask only whichever of these details is still missing:  
  • Business type  
  • How and where cards are accepted (fixed / mobile / remote)  
  • Built-in printer required  
  • Mobility (fixed vs mobile)  
  • Expected daily or monthly card transaction volume  
  • Rental budget (£/month)  
  • Contract preference (contract vs no contract)  
  • Required connectivity (Wi-Fi, Ethernet, 4G, Bluetooth)  
  • Required battery life (if mobile)
 
Recommendation Strategy  
Once you have all required details, recommend one or more devices from:  
{PRODUCT_CONTEXT} and {P_DESCRIPTIONS}
 When transaction info is available (txn_count + avg_value):
# - For each product get the details of transaction fees from {TR} and {P_DESCRIPTIONS}
#   - Use:  
#     - `FixedTransactionFees` (in pence like '10p')  
#     - `VariableTransactionFeess` (as percentage like '1.5%')
#   - Convert to:
#     - Fixed fee = pence / 100 (e.g., 10p = 0.10 GBP)
#     - Variable = percent / 100 (e.g., 1.5% = 0.015)

  - Formula:
    ```
   total_fee = (variable_rate * avg_txn_value) + (fixed_fee * txn_count)
   example:
   No of transactions	Tran fee	Average amount per transaction	Total Tran fee per transaction of 10 GBP	Total Tran fee for 1000 tran
      1000	             1.2%+10p	       10	                              0.22	                                        220
 	 	 	                                                                (10*1.2/100)+0.1	
    give me the final result. if asked for explanation then elaborate in a tabular format.                                                                           
For each recommended device:  
1. One-line rationale.  
2. Full specs in separate lines:  
   – Rental cost per month  
   – Connectivity  
   – Battery life  
   – Ideal for  
   – Transaction fees per transaction (fixed + variable)  
3. If transaction data is available, include “Estimated monthly transaction fee: £X.XX” in the same response.  
4. If purchase cost is available, calculate and display total monthly cost (purchase amortized, rental, transaction fees).
 
If you recommend more than one device, ask any additional clarifying question(s) needed to choose the best fit. If you recommend only one, present it confidently.
 
After recommending a Clover device, mention Clover premium upgrade options from:  
{PREMIUM}
 
Off-Topic Handling  
- For bank-related queries outside Cardnet POS recommendations:  
  “For other Bank inquiries, please contact the bank directly.”  
- For any other off-topic query:  
  “Let’s stay focused on Cardnet POS recommendations. Could you tell me more about your merchant’s needs?”
 
Unanswerable Questions  
If asked about Cardnet products and you cannot answer:  
“I am unable to answer that at this time. Would you like to enter your business details so our representative can reach out to you?”  
Then prompt for: business name, contact person, phone number, email.
 
Tone: Crisp, polite, professional. Avoid fluff. Be brief and clear.
"""

# ── Extract user contact info from chat ──────────────────────────────
def extract_customer_info(history):
    chat_text = " ".join(msg["content"] for msg in history if msg["role"] == "user")
    name_match = re.search(r"(?:my name is|contact name is)\s+([A-Za-z\s]+)", chat_text, re.I)
    phone_match = re.search(r"[0-9]*", chat_text)
    email_match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", chat_text)
    business_match = re.search(r"(?:business name is|my business is)\s+([A-Za-z0-9\s&,'\"-]+)", chat_text, re.I)

    return {
        "business_name": business_match.group(1).strip() if business_match else None,
        "customer_name": name_match.group(1).strip() if name_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "email": email_match.group(0) if email_match else None
    }

# ── Save chat with contact info ──────────────────────────────────────
def save_chat_history_to_file():
    chat_data = {
        "chat_start_time": st.session_state.chat_start_time,
        "chat": st.session_state.history[1:],  # exclude system
        "contact_info": extract_customer_info(st.session_state.history)
    }
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/chat_{st.session_state.chat_start_time}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

# ── Streamlit setup ──────────────────────────────────────────────────
st.set_page_config(page_title="🗨️ POS Recommender Chatbot", layout="centered")
st.title("🗨️ POS Recommender Chatbot")
st.markdown("Ask me about your merchant’s needs and I’ll recommend the perfect POS device.")

# ── Initialize session state ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
    ]
    st.session_state.chat_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.exiting = False

# ── Render chat history ──────────────────────────────────────────────
for msg in st.session_state.history[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Chat input + assistant response ──────────────────────────────────
if user_input := st.chat_input("Type your message…"):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.spinner("🤖 Thinking…"):
        response = client.chat.completions.create(
            model="models/gemini-1.5-flash-latest",
            messages=st.session_state.history,
            temperature=0.2,
        )
        reply = response.choices[0].message.content.strip()

    st.session_state.history.append({"role": "assistant", "content": reply})
    save_chat_history_to_file()
    st.rerun()

# ── Exit button logic ────────────────────────────────────────────────
if st.button("🚪 Exit Chat"):
    st.session_state.exiting = True

if st.session_state.exiting:
    info = extract_customer_info(st.session_state.history)
    missing = {k: v for k, v in info.items() if not v}

    if not missing:
        save_chat_history_to_file()
        st.success("✅ Your details are already saved. You may now close the window.")
        st.stop()
    else:
        st.warning("Before exiting, please complete the missing contact details:")
        if not info.get("business_name"):
            info["business_name"] = st.text_input("Business Name")
        if not info.get("customer_name"):
            info["customer_name"] = st.text_input("Customer Name")
        if not info.get("phone"):
            info["phone"] = st.text_input("Phone Number")
        if not info.get("email"):
            info["email"] = st.text_input("Email Address")

        if st.button("✅ Submit & Exit"):
            st.session_state.history.append({
                "role": "user",
                "content": f"My business is {info['business_name']}, my name is {info['customer_name']}, "
                           f"phone is {info['phone']}, email is {info['email']}."
            })
            save_chat_history_to_file()
            st.success("✅ Your details have been saved. You may now close the window.")
            st.stop()

