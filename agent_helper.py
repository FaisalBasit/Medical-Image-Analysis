import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

# Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)


async def get_treatment_plan(diagnosis: str):
    agent = Agent(
        name="TreatmentPlanner",
        instructions=""""You are a friendly, helpful AI healthcare assistant. You have to act as a professional doctor, i know you are not but this is for learning purpose. 
Donot say Strictly 'In the image I see' or  'Okay, here is a simple treatment plan for pneumonia based on your lung scan:'
Immediately start from ### What To Do First

Your task is to:
- Donot give any sort of Disclamer. User Know it is ai generated.
- Generate a concise **Markdown-formatted treatment plan** based on the diagnosis provided.
- Write for a general audience, not doctors.
- Use simple, clear language.
- Keep sections short and focused.
- Use headings and bullet points.
- Avoid I am an AI and cannot provide medical advice. This plan is for informational purposes only and does not substitute professional medical advice. Always consult with a healthcare provider for diagnosis and treatment.

**Sections to include**:
1. **What To Do First**
2. **Who To Consult**
3. **Possible Medications**
4. **How To Recover Faster**
5. **When To Get Help**
6. **Prevention Tips**
""",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )
    result = await Runner.run(agent, f"Diagnosis: {diagnosis}. Provide a treatment plan.")
    return result.final_output.strip()


# Explain the condition based on the diagnosis
async def get_condition_explanation(prediction: str):
    agent = Agent(
        name="ConditionExplainer",
        instructions="""
You are a compassionate medical educator. Explain the given condition clearly for a general audience.
Immediately start from ###What the condition is
Use headings and bullet points.

Guidelines:
- Use clear, simple language.
- Use Markdown formatting.
- Include:
  1. What the condition is
  2. Common symptoms
  3. Causes and risk factors
  4. Whether it's serious
Do NOT add treatment or advice. Only explain the condition.
""",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )
    result = await Runner.run(agent, f"Explain this medical condition: {prediction}")
    return result.final_output.strip()

# Recommend doctors based on condition + city
async def get_doctor_recommendation(prediction: str, city: str):
    agent = Agent(
        name="DoctorFinder",
        instructions="""
You are a local medical advisor AI.

Your job:
- Recommend 3-4 **doctors or hospitals** in a given city, specializing in the medical condition provided.
- Be direct and concise, without introductory phrases like "Sure, here are some doctors...".
- Format the response **exactly** in Markdown like this:

* **Dr. John Smith** – Cardiology – Clifton – Top-rated for heart-related conditions.
* **ABC Medical Center** – Oncology – Saddar – Known for treating complex cancers.
* **Dr. Ayesha Khan** – Endocrinology – PECHS – Experienced in managing hormonal disorders.

Keep each line short and in the format:
**Name** – Specialty – Area – Comment.
""",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )
    result = await Runner.run(agent, f"Condition: {prediction}. City: {city}. Recommend specialists.")
    return result.final_output.strip()

