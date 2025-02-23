import ast
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()


iteration = 0
dataset = ""

class State(TypedDict):
    prompt: str
    predictions: List[str]
    accuracy_score: Dict[str, Any]
    feedback: str
    Final_Report: Dict[str, Any]

def initialize_gemini_client():
    api_key = os.getenv("API_KEY")
    client = genai.Client(api_key=api_key)
    return client

client = initialize_gemini_client()


def initialize_state():
    state: State = {
        "prompt": "",
        "predictions": [],
        "feedback": "",
        "accuracy_score": {"macro-avg-f1_score": 0},
        "Final_Report": {}, 
    }
    return state

# Prompt Creation Agent
def prompt_creation_agent(state: State) -> State:
    global iteration
    print("+" * 100, f"Iteration: {iteration}", "+" * 100)

    temp = f"Iteration-{iteration}"
    
    state['Final_Report'][temp] = {
        "iteration": iteration,
        "prompt": state['prompt'],
        "feedback": state['feedback'],
        "accuracy_score": state['accuracy_score'],
        "predictions": state['predictions'],
    }

    global dataset

    prompt_template ="""
     Your Role & Responsibilities:
     As the Prompt Creation Agent, your task is to craft and refine the prompt used by the LLM for post classification.
     First Iteration: Create a new prompt from scratch, clearly explaining the classification task and defining each category.
     Subsequent Iterations: Refine the prompt based on feedback, improving language, expanding category definitions, or adding examples.
     Inputs:
     - Dataset: Review the dataset to understand the content and category representations.
     - Current Prompt: The prompt from the previous round, to be refined in subsequent iterations.
     - Feedback: Suggestions from the Feedback Creation Agent for improving clarity, category definitions, or examples.
     Template Variables:
     - Dataset: {Dataset}
     - Feedback: {Feedback}
     - Current Prompt: {Current_Prompt}

     Handling Feedback:

     - If {Feedback} is empty, create an initial prompt from scratch.
     - If {Feedback} is provided, refine the existing prompt based on the suggestions.
     Outputs:
     - A refined prompt that improves classification accuracy.
     - Return only the updated prompt without any prefixes.
     Important:
     - If Current Prompt is empty, create the initial prompt.
     - If {Feedback} is empty, generate a new prompt.
     - Only return the prompt for the classification agent.
     - Use the dataset to identify unique category values and mention them in the prompt. The classification agent should return results only from these values.
     """
    
    current_feedback = state['feedback']
    prompt = state['prompt']

    formatted_prompt = prompt_template.format(Dataset=dataset, Feedback=current_feedback, Current_Prompt=prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=formatted_prompt
    )

    print('=' * 100, "PROMPT CREATION AGENT RESULT", '=' * 100)
    print(response.text)
    state['prompt'] = response.text
    print('=' * 200)

    return state


# Classification Agent
def classification_agent(state: State) -> State:
    global dataset

    prompt_template = """Your Role & Responsibilities:

    You are the Classification Agent, responsible for classifying posts using the latest refined prompt from the Prompt Creation Agent.
    Use the refined prompt to guide the LLM in assigning categories to each post and record the predicted category based on the model's output.

    Inputs:

    - Refined Prompt: The latest refined prompt (or current prompt) from the Prompt Creation Agent.
    - Dataset: A collection of posts, including only the post_text for classification.

    Template Variables:

    - Refined Prompt: {Refined_Prompt}
    - Dataset: {Dataset}

    Outputs:

    - Predicted Labels: A list of predicted categories for each post, matched with the corresponding post_id.
    - Return only the result in JSON format.

    Example Output Format:

      () "post_id": 1,"predicted_category": "Sports")
    """
    current_prompt = state['prompt']
    formatted_prompt = prompt_template.format(Dataset=dataset[['post_id', 'post_text']], Refined_Prompt=current_prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=formatted_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )

    print('=' * 100, " CLASSIFICATION AGENT RESULT", '=' * 100)
    print(response.text)
    state['predictions'] = response.text
    print('=' * 200)
    return state

# Evaluation Agent
def evaluation_agent(state: State) -> State:
    global iteration
    prompt_template = """
    Your Role & Responsibilities:

    You are the Evaluation Agent, responsible for assessing the accuracy of the Classification Agent.
    Compare the predicted category labels with the correct ground truth labels.
    Compute key performance metrics:

    -   Confusion Matrix:   Displays how often the model correctly predicted each category and where errors occurred.
    -   F1 Scores (per category):   Measures precision and recall for each category.
    -   Macro-Average F1 Score:   The average of per-category F1 scores, serving as the main performance metric.

    The project is considered complete when the macro-average F1 score reaches 95%.

    Inputs:

    - Predicted Labels: Contains post_id and predicted category labels.
    - Actual Labels: Contains post_id, post_text, and ground truth category labels.

    Template Variables:

    - Predicted Labels: {Predicted_Labels}
    - Actual Labels: {Actual_Labels}

    Outputs:

    - Evaluation Metrics: Includes the confusion matrix, per-category F1 scores and the overall macro-average F1 score.
    - Return only the final accuracy value as an integer (macro-average F1 score × 100). Do not return any extra text like JSON .
    - The most important point is that you must return only the json, as you are an evaluation agent. Do not return the code.
    - Result must be like below format
    - The most importantly labels a add predicted and actual labes is needed for each catogory in confusion metircs. Return the confusion metics as dictionary
    - No need to retunr the final value , return only the json
    - In Calulation_and_Formula return the calculation and Formula used for calculation, Detailed explation needed
    Output Format:

    The final output should be a json,
    e.g.:

    ("Calulation_and_Formula":"Calculation and Fomula used for calculate",
  "confusion-matrix": [
    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  ],
  "f1-scores": (
    "Description or Definition or Background Context": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Question": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Request Explanation": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Resource Request": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Advice Request": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Seeking Relationship": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Clarification Request": (
      "precision": 0.25,
      "recall": 0.333,
      "f1-score": 0.285
    ),
    "Task Request": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Sharing Information/Observation": (
      "precision": 0.166,
      "recall": 0.833,
      "f1-score": 0.277
    ),
    "Informal post (other)": (
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0
    ),
    "Praise/Gratitude": (
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0
    ),
    "Ask for clarification about the topic / scope": (
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0
    )
  ),
  "macro-avg-f1_score": 0.713)
    """
    global dataset
    Predicted_Labels = state['predictions']

    formatted_prompt = prompt_template.format(Predicted_Labels=Predicted_Labels, Actual_Labels=dataset)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=formatted_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )
    string = response.text
    cleaned_string = string.replace("```json", "").replace("```", "").strip()[::]
    metrics_data_dict = ast.literal_eval(cleaned_string)

    print('=' * 100, " --EVALUATION AGENT RESULT--", '=' * 100)
    print(metrics_data_dict)
    state['accuracy_score'] = metrics_data_dict
    print(state['accuracy_score']['macro-avg-f1_score'] * 100)
  
    temp = f"Iteration-{iteration}"
    state['Final_Report'][temp] = {
        "iteration": iteration,
        "prompt": state['prompt'],
        "feedback": state['feedback'],
        "accuracy_score": state['accuracy_score'],
        "predictions": state['predictions'],
    }

    iteration += 1  
    return state



# Feedback Creation Agent
def feedback_creation_agent(state: State) -> State:
    prompt_template = """

    Your Role & Responsibilities:

    You are the Feedback Creation Agent, responsible for analyzing the results from the Evaluation Agent and providing suggestions to improve the classification prompt.

    Inputs:

    - Current Prompt: The latest prompt from the Prompt Creation Agent.
    - Predicted Labels: Predicted labels from the Classification Agent.
    - Actual Labels: Ground-truth labels from the dataset (includes post_id, post_text, and category. Consider `category` as the actual label).
    - Evaluation Metrics: Evaluation results from the Evaluation Agent.

    Template Variables:

    - Current_Prompt: {CurrentPrompt}
    - Predicted_Labels: {PredictedLabels}
    - Actual_Labels: {ActualLabels}
    - Evaluation_Metrics: {EvaluationMetricsValue}

    Handling Missing Inputs:

    - If any of the inputs ({CurrentPrompt}, {PredictedLabels}, {ActualLabels}, {EvaluationMetricsValue}) is missing or None, return "None" explicitly.
    - If all required inputs are present, analyze errors and generate precise feedback.

    Outputs:

    - Feedback Report: Actionable suggestions to improve the prompt, such as:
      - Refining definitions of commonly confused categories.
      - Adding keywords or examples for better classification.
      - Reworking ambiguous parts of the prompt.

    Important:

    - Avoid prefacing responses with "Okay" or similar words.
    """
    print('=' * 100, " FEEDBACK AGENT RESULT", '=' * 100)
    Current_Prompt = state['prompt']
    Predicted_Labels = state['predictions']
    global dataset
    MetricsValue = state['accuracy_score']['macro-avg-f1_score']

    formatted_prompt = prompt_template.format(CurrentPrompt=Current_Prompt, PredictedLabels=Predicted_Labels, ActualLabels=dataset, EvaluationMetricsValue=MetricsValue)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=formatted_prompt
    )

    print(response.text)
    state['feedback'] = response.text
    print('=' * 200)
    return state



# Create the graph
def create_graph(data_set):
    graph_builder = StateGraph(State)
    global dataset
    dataset = data_set
    graph_builder.add_node("prompt_creation_agent", prompt_creation_agent)
    graph_builder.add_node("classification_agent", classification_agent)
    graph_builder.add_node("evaluation_agent", evaluation_agent)
    graph_builder.add_node("feedback_creation_agent", feedback_creation_agent)

    graph_builder.add_edge(START, "prompt_creation_agent")
    graph_builder.add_edge("prompt_creation_agent", "classification_agent")
    graph_builder.add_edge("classification_agent", "evaluation_agent")
    graph_builder.add_conditional_edges(
        "evaluation_agent",
        lambda state: "feedback_creation_agent" if (state['accuracy_score']['macro-avg-f1_score'] * 100) < 95 else END,
    )

    graph_builder.set_entry_point("prompt_creation_agent")

    graph = graph_builder.compile()
    return graph


# Evaluate custom prompt
def evaluate_prompt(state, data_set, custom_prompt):
    global dataset
    dataset = data_set
    print("=" * 200)
    print("=" * 100, " CUSTOM PROMPT ", "=" * 100)
    print(custom_prompt)
    print("=" * 200)
    classification_agent(state)
    evaluation_agent(state)
    feedback_creation_agent(state)
    return [state['accuracy_score']['macro-avg-f1_score'], state['prompt']]