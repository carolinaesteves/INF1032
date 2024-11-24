import plotly.graph_objects as go

# Constants     
gridcolor = 'rgba(225,226,220,1)'

xaxis = dict(title="<b></b>",
            titlefont=dict(family='Arial', size=18, color='rgb(00,00,00)'),  # size was 16
            tickmode='auto',
            tickformat="%b-%Y")

yaxis = dict(title="<b></b>",
            titlefont=dict(family='Arial', size=18, color='rgb(00,00,00)'),  # size was 16
            tickformat=".2%",
            overlaying='y2',
            showgrid=True,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='Black',
            gridcolor=gridcolor)
layout = go.Layout(title=dict(text='',
                            x=0.5,
                            font=dict(family='arial', size=20),
                            xanchor='center',
                            yanchor='top'),
                hovermode="x",
                font=dict(family='arial', size=13.5),  # size was 12
                legend=dict(orientation="h", font=dict(size=14.5)),  # size was 12
                showlegend=True,
                xaxis=xaxis,
                yaxis=yaxis,
                plot_bgcolor='rgba(0,0,0,0)',
                )

def plot(list_preds,labels,y_test):
    fig = go.Figure()
    fig.layout = layout
    fig.layout.yaxis.tickformat = ".2%"

    for i in range(0,len(list_preds)):
        fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=list_preds[i]/100,
                mode="lines",
                name = labels[i]
        ))
    
    return fig
