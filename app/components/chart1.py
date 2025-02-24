
import plotly.graph_objects as go
import streamlit as st
def display_accuracy_chart(final_report):
    print("++"*100,"Line plot","++"*100,)

    #Extract data for Line plot
    f1_scores_list = [0]
    for iteration_key, iteration_data in final_report.items():  
        f1_scores_list.append(iteration_data['accuracy_score']['macro-avg-f1_score']*100)
    iterations_list=[i for i in range(len(f1_scores_list))]

    #Plotting Line Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations_list,
        y=f1_scores_list,
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='mediumslateblue', width=2),
        marker=dict(symbol='circle', size=8, color='gold', line=dict(color='black', width=1))
    ))
    fig.update_layout(
        title=dict(
            text='F1 Score Over Iterations',
            font=dict(size=24, color='darkblue', family='Arial', weight='bold'),
            x=0.5,  
            xanchor='center' 
        ),
        xaxis=dict(
            title='Iteration',
            title_font=dict(size=18, color='darkgreen'),
            tickfont=dict(size=14, color='darkred'),
            gridcolor='lightgray',
            gridwidth=1,
            showgrid=True
        ),
        yaxis=dict(
            title='F1 Score',
            title_font=dict(size=18, color='darkgreen'),
            tickfont=dict(size=14, color='darkred'),
            range=[0, 100],  
            dtick=5,  
            gridcolor='lightgray',
            gridwidth=1,
            showgrid=True
        ),
        legend=dict(
            x=1,  
            y=1,
            bgcolor='lightyellow',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode='x unified', 
        template='plotly_white', 
        plot_bgcolor='white',  
        paper_bgcolor='white'  
    )
    st.plotly_chart(fig)
