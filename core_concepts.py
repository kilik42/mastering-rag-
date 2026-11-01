from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain

def demo_batch_execution():
    """Demonstrates batch execution using LCEL and Runnables."""

    prompt = ChatPromptTemplate.from_template(
        ChatPromptTemplate.from_template("translate to french: {text}")
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #batch - run with multiple inputs
    texts = [
        {"text": "Hello, how are you?"},
        {"text": "Good morning!"},
        {"text": "See you later."}
    ]

    # for t in texts:
    #     result = chain.invoke(t)
    #     print(f"Response: {result}")

    results  = chain.batch(texts)
    for text, result in zip(texts, results):
        print(f"Input: {text['text']}, Response: {result}")

    return chain


def demo_streaming():
    """Demonstrate streaming for real-time output."""

    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about: {topic}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")

    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)


def demo_schema_inspection():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text: {text}"
    )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for the product: {product} targeting {audience}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Generated tagline: {result}")
    return chain

if __name__ == "__main__":
    demo_basic_chain()
    demo_batch_execution()
    demo_streaming()
    demo_schema_inspection()
    exercise_first_chain()