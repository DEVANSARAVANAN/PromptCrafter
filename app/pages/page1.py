
import streamlit as st
from app.components.chart1 import display_accuracy_chart
from app.components.chart2 import pie
from app.components.chart3 import bar_chart
from app.components.chart4 import heat_map
from app.utils.model import  create_graph, initialize_state

def run_page1(dataset):    
    state = initialize_state()
    graph = create_graph(dataset)   
    
    while (state['accuracy_score']['macro-avg-f1_score'] * 100) < 95:    
        state = graph.invoke(state, {"recursion_limit": 30})
    
    st.write("\nFinal Report:")
    display_accuracy_chart(state["Final_Report"])
    heat_map(state['accuracy_score'])
    bar_chart(state['accuracy_score'])
    pie(state["Final_Report"], dataset)
    st.write("Macro Average of F1 Score:",state['accuracy_score']['macro-avg-f1_score']*100)
    

        

