import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve



def eval_curves(fpr, tpr, auc, y_test, y_prob):

    #  ROC-AUC Curve
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    
    fig.add_shape(
        type='line', line=dict(dash='dash'), 
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(autosize=True,  
                    margin=dict(l=30, r=30, b=30, t=30))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    #  Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:,1])

    fig2 = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AUC={auc})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )

    fig2.add_shape(
        type='line', line=dict(dash='dash'), 
        x0=0, x1=1, y0=0, y1=1
    )
    fig2.update_layout(autosize=True,  
                    margin=dict(l=30, r=30, b=30, t=30))
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    fig2.update_xaxes(constrain="domain")

    # Confusion Matrix
    
    # conf_matrix_array = conf_matrix_array.tolist()
    # x = ['No Heart Failure', 'Heart Failure']
    # y =  ['No Heart Failure', 'Heart Failure']

    # # change each element of z to type string for annotations
    # z_text = [[str(y) for y in x] for x in conf_matrix_array]

    # # set up confusion matrix figure 
    # fig3 = ff.create_annotated_heatmap(conf_matrix_array, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # # add title
    # fig3.update_layout(title_text='<i><b>Confusion matrix</b></i>',
    #                 xaxis = dict(title='Predicted value'),
    #                 yaxis = dict(title='Actual value')
    #                 )

    # # add custom xaxis title
    # fig3.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=0.5,
    #                         y=-0.15,
    #                         showarrow=False,
    #                         text="Predicted value",
    #                         xref="paper",
    #                         yref="paper"))

    # # add custom yaxis title
    # fig3.add_annotation(dict(font=dict(color="black",size=14),
    #                         x=-0.35,
    #                         y=0.5,
    #                         showarrow=False,
    #                         text="Real value",
    #                         textangle=-90,
    #                         xref="paper",
    #                         yref="paper"))

    # # adjust margins to make room for yaxis title
    # fig3.update_layout(margin=dict(t=50, l=200))

    # # add colorbar
    # fig3['data'][0]['showscale'] = True

    return fig, fig2