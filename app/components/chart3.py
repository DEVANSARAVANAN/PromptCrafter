import plotly.express as px
import pandas as pd
import streamlit as st

def bar_chart(data):
    print("++"*100,"Bar Chart","++"*100)
    metrics = []
    for class_name, scores in data['f1-scores'].items():
        metrics.append({
            'Class': class_name,
            'Precision': scores['precision'],
            'Recall': scores['recall'],
            'F1 Score': scores['f1-score']
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.melt(id_vars=['Class'], var_name='Metric', value_name='Score')
    fig = px.bar(metrics_df,
                x='Class',
                y='Score',
                color='Metric',
                barmode='group',
                text='Score',
                labels={'Class': 'Class', 'Score': 'Score', 'Metric': 'Metric'})
    fig.update_traces(textposition='outside', texttemplate='%{y:.2f}')
    fig.update_layout(
        title={
            'text': 'Comparison of Precision, Recall, and F1 Score',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=14)
        },
        xaxis_title='Class',
        yaxis_title='Score',
        width=1200,
        height=800,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10))
    )
    st.plotly_chart(fig)