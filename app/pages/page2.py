
import streamlit as st
from app.utils.model import  initialize_state,evaluate_prompt
from app.components.chart1 import display_accuracy_chart
from app.components.chart2 import pie
from app.components.chart3 import bar_chart
from app.components.chart4 import heat_map

def run_page2(dataset,custom_prompt):
    state = initialize_state()
    evaluate_prompt(state,dataset,custom_prompt)
    st.write(state['accuracy_score']['macro-avg-f1_score'] * 100)
    
    #Plottings
    display_accuracy_chart(state["Final_Report"])
    pie(state["Final_Report"], dataset)
    bar_chart(state['accuracy_score'])
    heat_map(state['accuracy_score'])
    return state['accuracy_score']['macro-avg-f1_score'] * 100
    
    

        

