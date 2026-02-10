# =====================================
# LangChain Practice Project (NO API)
# Uses FakeListLLM (OFFICIAL)
# =====================================

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch
from langchain_core.language_models.fake import FakeListLLM

# =====================================
# Fake LLM Setup (NO INTERNET)
# =====================================

fake_responses = [
    "This is a fake response generated for learning purposes."
]

llm_fast = FakeListLLM(responses=fake_responses)
llm_smart = FakeListLLM(responses=fake_responses)

parser = StrOutputParser()

# =====================================
# Input Text
# =====================================

text = """
LangChain is a framework that helps developers build applications
powered by large language models. It supports sequential, parallel,
and conditional workflows.
"""

# =====================================
# PROBLEM 1: SEQUENTIAL CHAIN
# =====================================

print("\n========== PROBLEM 1: SEQUENTIAL CHAIN ==========\n")

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 3–4 lines:\n{text}"
)

keypoints_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Extract 5 key points from the summary:\n{summary}"
)

questions_prompt = PromptTemplate(
    input_variables=["keypoints"],
    template="Generate 3 interview questions from these key points:\n{keypoints}"
)

summary = parser.invoke(llm_fast.invoke(summary_prompt.format(text=text)))
keypoints = parser.invoke(llm_fast.invoke(keypoints_prompt.format(summary=summary)))
questions = parser.invoke(llm_fast.invoke(questions_prompt.format(keypoints=keypoints)))

print("SUMMARY:\n", summary)
print("\nKEY POINTS:\n", keypoints)
print("\nINTERVIEW QUESTIONS:\n", questions)

# =====================================
# PROBLEM 2: PARALLEL CHAIN
# =====================================

print("\n========== PROBLEM 2: PARALLEL CHAIN ==========\n")

parallel_chain = RunnableParallel(
    Notes=PromptTemplate(
        input_variables=["text"],
        template="Write simple notes:\n{text}"
    ) | llm_fast | parser,

    Quiz=PromptTemplate(
        input_variables=["text"],
        template="Create 5 MCQ questions:\n{text}"
    ) | llm_smart | parser,

    Use_Cases=PromptTemplate(
        input_variables=["text"],
        template="Give 3 real-world use cases:\n{text}"
    ) | llm_fast | parser
)

parallel_output = parallel_chain.invoke({"text": text})

print("NOTES:\n", parallel_output["Notes"])
print("\nQUIZ:\n", parallel_output["Quiz"])
print("\nUSE CASES:\n", parallel_output["Use_Cases"])

# =====================================
# PROBLEM 3: CONDITIONAL CHAIN
# =====================================

print("\n========== PROBLEM 3: CONDITIONAL CHAIN ==========\n")

def detect_domain(text):
    if "student" in text.lower() or "learning" in text.lower():
        return "education"
    return "business"

education_chain = RunnableParallel(
    Study_Notes=PromptTemplate(
        input_variables=["text"],
        template="Create study notes:\n{text}"
    ) | llm_fast | parser,

    Practice_Questions=PromptTemplate(
        input_variables=["text"],
        template="Create practice questions:\n{text}"
    ) | llm_fast | parser
)

business_chain = RunnableParallel(
    Business_Summary=PromptTemplate(
        input_variables=["text"],
        template="Create business summary:\n{text}"
    ) | llm_smart | parser,

    Risk_Analysis=PromptTemplate(
        input_variables=["text"],
        template="Analyze business risks:\n{text}"
    ) | llm_smart | parser
)

conditional_chain = RunnableBranch(
    (lambda x: detect_domain(x["text"]) == "education", education_chain),
    business_chain  # default branch
)

conditional_output = conditional_chain.invoke({"text": text})

print("FINAL CONDITIONAL OUTPUT:\n", conditional_output)
