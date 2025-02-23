import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json


def pie(final_report, dataset):
    print("++"*100,"Pie Chart","++"*100)
    category_frequencies = {}
    sorted_iteration_keys = sorted(final_report.keys(), key=lambda x: int(x.split('-')[-1]), reverse=True)

    for iteration_key in sorted_iteration_keys:
        iteration_data = final_report[iteration_key]
        predictions = iteration_data.get('predictions', '')
        if isinstance(predictions, str):
            try:
                json_start = predictions.find("[")
                json_end = predictions.rfind("]") + 1
                predictions_json = predictions[json_start:json_end]
                predictions_list = json.loads(predictions_json)
            except (json.JSONDecodeError, ValueError):
                print(f"Skipping {iteration_key} due to JSON parsing error.")
                continue
        elif isinstance(predictions, list):
            predictions_list = predictions  
        else:
            print(f"Skipping {iteration_key} due to unexpected data format.")
            continue


        category_counts = {}
        for prediction in predictions_list:
            category = prediction.get("predicted_category")
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
        category_frequencies[iteration_key] = category_counts
    num_iterations = max(len(category_frequencies), 1)  

    fig = make_subplots(
        rows=num_iterations,
        cols=2,
        subplot_titles=["Predicted Data", "Actual Data"],
        specs=[[{"type": "domain"}, {"type": "domain"}] for _ in range(num_iterations)]
    )

    for i, (iteration_key, category_counts) in enumerate(category_frequencies.items(), start=1):
        labels = list(category_counts.keys())
        values = list(category_counts.values())
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.3, title=iteration_key), row=i, col=1)



    if not dataset.empty and "category" in dataset.columns:
        category_counts_dataset = dataset["category"].value_counts()
        fig.add_trace(go.Pie(labels=category_counts_dataset.index, values=category_counts_dataset.values, hole=0.3, title="Actual Data"), row=1, col=2)
    else:
        print("Dataset is empty or missing 'category' column. Skipping actual data visualization.")

    fig.update_layout(
        title_text="Comparison of Category Frequencies (Predicted Data & Actual Data)",
        showlegend=True,
        height=400 * num_iterations, 
        template="plotly_white"
    )
    st.plotly_chart(fig)
