**Agentic AI Tool for Automated Prompt Building and Iteration for Text Classification
Overview**

This project aims to develop a Generative AI (GenAI) system that automatically generates and iteratively refines prompts for text classification tasks. The system optimizes the performance of a Large Language Model (LLM) by enhancing prompt quality, ensuring accurate classification of posts into predefined categories.

**Key Features:**

  1.Multi-Agent Collaboration: Utilizes four agents (Prompt Creation, Classification, Evaluation, and Feedback Creation) to iteratively refine prompts.
  
  2.High Accuracy: Achieves a 95% macro-average F1 score for text classification.
  
  3.Visualization: Provides interactive charts and graphs for performance analysis.
  
  4.Custom Prompt Evaluation: Allows users to test custom prompts and evaluate their performance.


**Table of Contents**
- Installation
- Usage
- Workflow
- Agents
- Model Analysis & Optimization
- Visualization
- Output



**Installation Prerequisites:**

- Python 3.8 or higher
- Streamlit
- LangGraph
- Google GenAI
- Pandas
- Plotly 

**Steps Clone the repository:**

- git clone https://github.com/DEVANSARAVANAN/PromptCrafter.git

- cd PromptCrafter


**Install the required dependencies:**


- pip install -r requirements.txt


**Set up Google GenAI API:**

Obtain an API key from Google GenAI.


- Create a .env file in the root directory of the project and add your API key:

  **.env file:**

  GOOGLE_GENAI_API_KEY=your-api-key


**Run the application:**

  streamlit run app.py


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Usage:**

**1.Run the Streamlit App:**

  streamlit run app.py


**2.Upload Dataset:**

**Upload a CSV or XLSX file containing:**
  
  
  post_id: Unique identifier for each post.
  
  post_text: Content of the post.
  
  category: Label for the post (e.g., "Description," "Definition," "Background Context").


**3.Choose Data Processing Option:**
  
  **Unprocessed Data:** Preprocess the dataset for better classification.
  
  **Processed Data:** Use the dataset as-is.


**4.Start Optimization:**

  The system will iteratively refine the prompt until the F1 score reaches 95%.


**5.Evaluate Custom Prompts:**

  Input a custom prompt and evaluate its performance.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Workflow:**

**1.Data Upload and Processing:**

  Users upload a dataset via the Streamlit interface.
  
  The dataset is preprocessed to improve classification accuracy.

**2.Prompt Optimization:**

**3.The multi-agent system iteratively refines the prompt:**
  
  - Prompt Creation Agent: Generates or refines the prompt.
  
  - Classification Agent: Classifies posts using the prompt.
  
  - Evaluation Agent: Calculates the F1 score.
  
  - Feedback Creation Agent: Provides feedback to improve the prompt.

**4.Custom Prompt Evaluation:**

  Users can input a custom prompt and evaluate its performance.

**5.Visualization:**

  The application displays accuracy charts and pie charts for performance analysis.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Agents:**

**1.Prompt Creation Agent:**

  Generates and refines prompts for the LLM.
  
  Inputs: Dataset, previous prompts, feedback report.
  
  Outputs: Optimized prompt.

**2.Classification Agent:**

  Uses the LLM to classify posts based on the prompt.
  
  Inputs: Refined prompt, dataset (post_text).
  
  Outputs: Predicted labels.

**3.Evaluation Agent:**

  Measures classification performance.
  
  Inputs: Predicted labels, ground truth labels.
  
  Outputs: Evaluation metrics (confusion matrix, F1 scores).

**4.Feedback Creation Agent:**

  Analyzes misclassified posts and generates recommendations for prompt improvement.
  
  Inputs: Current prompt, predicted vs. actual labels, evaluation metrics.

  Outputs: Feedback report.
  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Model Analysis & Optimization:**
Testing Model Accuracy: The model reaches the target accuracy within 1 to 12 iterations.

Reached Target in Different Iterations:

  - Best case: Target reached in 1 iteration.
  
  - Worst case: Target reached in 18 iterations.

Modified Model Parameters: Adjusting the temperature parameter to 0.1 improved performance.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Visualization:**

**1.Confusion Matrix Heat Map:**

  - Displays predicted vs. actual labels across categories.
  
  - Comparison of Precision, Recall, and F1 Score:
  
  - Bar graph comparing performance metrics.

**2.Comparison of Category Frequencies:**

  Pie chart comparing predicted vs. actual data for each iteration.

**3.JSON File Structure:**

  Hierarchical structure of the JSON file for analysis.
  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Output Backend Initial Iteration:**
  
  - Initial Prompt Creation Agent Output.
  
  - Initial Classification Agent Output.
  
  - Initial Evaluation Agent Output.
  
  - Initial Feedback Agent Output.

**Final Iteration:**

  - Final Iteration Prompt Creation Agent Output.
  
  - Final Iteration Classification Agent Output.
  
  - Final Iteration Evaluation Agent Output.

**Frontend User Guidance:**

Step-by-step instructions for using the tool.

  - **Final Report and Final Prompt:**

Displays the final report and prompt after optimization.

  - **F1 Score Over Iterations:**

Graph showing F1 score progression across iterations.

  - **Confusion Matrix Heatmap:**

Heatmap showing predicted vs. actual labels.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


