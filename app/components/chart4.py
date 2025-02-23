
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st

def heat_map(data):
    print("++"*100,"Heat Map","++"*100)  
    classes = list(data['confusion-matrix'].keys())
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
   
    for i, actual_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            if predicted_class in data['confusion-matrix'][actual_class]['predicted']:
                conf_matrix[i, j] = data['confusion-matrix'][actual_class]['predicted'][predicted_class]

    conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    fig = px.imshow(conf_df,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=True,
                color_continuous_scale='Blues')
    fig.update_layout(
    title={
        'text': 'Confusion Matrix Heatmap',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=14)
    },
    xaxis_title='Predicted',
    yaxis_title='Actual',
    width=1000,
    height=800,
    font=dict(size=10),
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10))
    )   
    st.plotly_chart(fig)