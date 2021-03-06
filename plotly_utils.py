# import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.io as pio
from plotly.graph_objs import Figure
import plotly.figure_factory as ff
from plotly import tools
import cufflinks as cf
cf.go_offline()
import calc_utils as cx
import re
# import numpy as np
# import networkx as nx
# import community
# import seaborn as sns
# from python_utils import matrix_utils as mx

# offline.init_notebook_mode()

'''
colorscale options
'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
'Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
'''

# colors rgb values
rgb = [
    (75, 175, 79),
    (243, 65, 53),
    (253, 86, 33),
    (253, 192, 6),
    (14, 166, 136)
    ]

# from paired https://plot.ly/pandas/colorlover/
rgb2 = [
    (31, 120, 180),
    (51, 160, 44),
    (227, 26, 28),
    (255, 127, 0)
    ]


def color(rgb, opacity):
    return 'rgba({0}, {1}, {2}, {3})'.format(rgb[0], rgb[1], rgb[2], opacity)


def get_axis(columns, type, horizontal=False):
        data = []
        idx = 0
        dash = ['dash', 'dot', 'dashdot']

        rgb = rgb2

        for (x, y, label) in columns:
            trace = 'trace_{0}'.format(idx)
            if type == 'bar':
                trace = go.Bar(
                    x=x,
                    y=y,
                    name=label,
                    marker=dict(
                        color=color(rgb[0], 0.5),
                        line=dict(
                            color=color(rgb[0], 1.0),
                            width=1)
                            )
                        )
            elif type == 'lines':
                trace = go.Scatter(
                    x=x,
                    y=y,
                    name=label,
                    mode='lines+markers',
                    line=dict(width=4)
                    )
            if horizontal:
                trace['orientation'] = 'h'

            if len(columns) < len(rgb):
                trace['marker']=dict(
                    color=color(rgb[idx], 0.5),
                    line=dict(
                        color=color(rgb[idx], 1.0),
                        width=3)
                )

            if idx >=1:
                trace['line']['dash'] = dash[idx - 1]

            idx += 1
            data.append(trace)
        # print(len(data))
        return data


def draw_chart(title, data, **kwargs):

    fname = title + '.html'

    if 'layout' in kwargs:
        data_dict = {"data": data, "layout": kwargs['layout']}
    else:
        data_dict = {"data": data}

    if 'image' in kwargs:
        offline.plot(data_dict, image=kwargs['image'], image_filename=title, filename=fname)
    else:
        offline.plot(data_dict, filename=fname)

    # orca export formats: 'png', 'jpeg', 'webp', 'svg', 'pdf', or 'eps'.
    if 'format' in kwargs:
        figure = Figure(data=data, layout=kwargs['layout'])
        pio.write_image(figure, title + '.' + kwargs['format'])

    if 'clickable' in kwargs:
        plot_div = offline.plot(data_dict, output_type='div', include_plotlyjs=True)

        # Get id of html div element that looks like
        # <div id="301d22ab-bfba-4621-8f5d-dc4fd855bb33" ... >
        res = re.search('<div id="([^"]*)"', plot_div)
        div_id = res.groups()[0]

        # Build JavaScript callback for handling clicks
        # and opening the URL in the trace's customdata 
        js_callback = """
        <script>
        var plot_element = document.getElementById("{div_id}");
        plot_element.on('plotly_click', function(data){{
            console.log(data.points[0]);
            var point = data.points[0];
            if (point) {{
                console.log(point.text);
                window.open(point.text);
            }}
        }})
        </script>
        """.format(div_id=div_id)

        # Build HTML string
        html_str = """
        <html>
        <body>
        {plot_div}
        {js_callback}
        </body>
        </html>
        """.format(plot_div=plot_div, js_callback=js_callback)

        # Write out HTML file
        with open('{}.html'.format(title), 'w') as f:
            f.write(html_str)
        

def pandas_bar_chart(x_col, y_col, title):

    data = [
        go.Bar(
            x=x_col,
            y=y_col,
            name=title,
            marker=dict(
                color=color(rgb[0], 0.5),
                line=dict(
                    color=color(rgb[0], 1.0),
                    width=1)
                    )
                )
        ]

    layout = go.Layout(
        title=title,
        width=1600,
        # barmode=kwargs['layout'],
        margin=go.Margin(
            l=250,
            r=30,
            b=50,
            t=30,
            pad=20
        )
        )

    draw_chart(title, data, layout=layout)


def bar_chart(columns, title, horizontal=False, **kwargs):

    data = get_axis(columns, horizontal)

    if 'layout' in kwargs:
        layout = go.Layout(
            title=title,
            width=1600,
            barmode=kwargs['layout'],  # layout mode: 'stack' | 'group'
            margin=go.Margin(
                l=250,
                r=30,
                b=50,
                t=30,
                pad=20
            )
            )
        draw_chart(title, data, layout=layout, image='svg')
    else:
        draw_chart(title, data, image='svg')


def bar_group_stacked_chart(columns, title, horizontal=False):

    data = get_axis(columns, horizontal)

    # for column in data:
        # column['width'] = 500

    red = dict(
        color=color(rgb[1], 0.5),
        line=dict(
            color=color(rgb[1], 1.0),
            width=1)
    )

    green = dict(
        color=color(rgb[0], 0.5),
        line=dict(
            color=color(rgb[0], 1.0),
            width=1)
    )
    # data[0]['offset'] = 0
    # data[0]['base'] = 0
    # data[1]['offset'] = -0.5
    # data[2]['offset'] = 1
    # data[3]['offset'] = 0.5

    # for column in data[:2]:
    #     # column['base'] = 0.0
    #     # column['offset'] = 0.0
    #     column['marker'] = red
    #
    # for column in data[2:]:
    #     # column['offset'] = -0.5
    #     column['marker'] = green
    #
    # print(data)

    layout = go.Layout(
        title=title,
        width=1600,
        barmode='group',
        # bargap=50,
        margin=go.Margin(
            l=250,
            r=30,
            b=50,
            t=30,
            pad=20
        )
        )

    fig = tools.make_subplots(rows=2, cols=1, shared_yaxes=True)


    draw_chart(title, data, layout=layout, image='svg')

    # print(data)


def stacked_bar_chart(columns, title):

    data = get_axis(columns, horizontal=False)

    layout = go.Layout(
        title=title,
        # autosize=False,
        width=1600,
        # height=500,
        barmode='stack',
        margin=go.Margin(
            l=250,
            r=30,
            b=50,
            t=30,
            pad=20
        ),
        # plot_bgcolor=color((0, 0, 0), 0.05),
        # paper_bgcolor=color((0, 0, 0), 0.05)
    )

    draw_chart(title, data, layout=layout, image='svg')

def scatter_chart_line(title, x, y1, y2, line):

    trace0 = go.Scatter(
        x = x,
        y = y1,
        mode = 'markers',
        name = 'markers'
    )
    trace1 = go.Scatter(
        x = x,
        y = y2,
        mode = 'markers',
        name = 'markers'
    )

    trace2 = go.Scatter(
                  x=x,
                  y=line,
                  mode='lines',
                  marker=go.Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

    data = [trace0, trace1, trace2]

    layout = go.Layout(
        # autosize=False,
        # width=2500,
        # height=2000,
        title=title,
        xaxis=dict(ticks=''),
        yaxis=dict(ticks='', tickfont=dict(size=12)),
        margin=dict(b=40, l=120, r=5, t=40, pad=5)
    )

    draw_chart(title, data, layout=layout, image='png')


def scatter_chart(title, x, y, labels, text, colorscale='viridis', format='pdf'):

    data = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='markers',
        marker=dict(
            size=8,
            color=labels,
            colorscale=colorscale
        ),
        text=text
    )

    layout = go.Layout(
        # autosize=False,
        # width=2500,
        # height=2000,
        title=title,
        xaxis=dict(ticks=''),
        yaxis=dict(ticks='', tickfont=dict(size=12)),
        margin=dict(b=40, l=120, r=5, t=40, pad=5)
    )

    draw_chart(title, data, layout=layout, clickable=True)


def network_chart(df, title, layout='spring'):

    g = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])

    # create the layout for the nodes
    if layout == 'spring':
        pos = nx.spring_layout(g, iterations=1)
    elif layout == 'circular':
        pos = nx.circular_layout(g, scale=50)
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(g, iterations=5)
    elif layout == 'random':
        pos = nx.random_layout(g)

    # for i in g.nodes():
    #     print(layout[i])

    # community detection
    parts = community.best_partition(g)

    # node degree centrality
    dc = mx.get_degree_centrality(g)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.25, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in g.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        textposition='top left',
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in g.nodes():
        x, y = pos[node]
        node_trace['text'] += (node,)
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['color'] += (parts.get(node),)
        size = dc.loc[dc['node'] == node, 'degree_centrality'].item()
        node_trace['marker']['size'] += (size / 3)

    # for ix, adjacencies in enumerate(nx.generate_adjlist(g)):
        # print(node, adjacencies)
        # node_trace['marker']['size'].append(len(adjacencies))
        # node_trace['marker']['color'].append(len(adjacencies))
        # node_info = '# of connections: '+str(len(adjacencies))
        # node_trace['text'].append(node_info)


    data = [edge_trace, node_trace]

    layout = go.Layout(
        title=title,
        titlefont=dict(size=16),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

    draw_chart(title, data, layout=layout, image='png')


def draw_heatmap(df, columns, title, colorscale, format='pdf'):
    # sns.heatmap(df, annot=True)
    data = [
        go.Heatmap(
            x=df[columns[0]],
            y=df[columns[1]],
            z=df[columns[2]],
            colorscale=colorscale
        )
    ]

    layout = go.Layout(
        autosize=False,
        width=2500,
        height=2000,
        title=title,
        xaxis=dict(ticks='', nticks=len(df[columns[0]])),
        yaxis=dict(ticks='', nticks=len(df[columns[1]]), tickfont=dict(size=12)),
        margin=dict(b=40, l=120, r=5, t=40, pad=5)
    )

    draw_chart(title, data, layout=layout, format=format)


def draw_lines_chart(columns, title, format='pdf', **kwargs):

    data = get_axis(columns, 'lines', horizontal=False)

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1200,
        title=title,
        legend=dict(orientation="h", font=dict(size=18)),
        xaxis=dict(ticks='', tickfont=dict(size=18)),
        yaxis=dict(ticks='', tickfont=dict(size=18)),
        margin=dict(b=40, l=100, r=5, t=40, pad=5)
    )

    if 'annotations' in kwargs:
        layout['annotations'] = [dict(
            x=kwargs['x'],
            y=kwargs['y'],
            xref='x',
            yref='y',
            text=kwargs['text'],
            showarrow=False,
            font=dict(size=14),
            align='left',
            # arrowhead=2,
            # arrowsize=1,
            # arrowwidth=2,
            # arrowcolor='#636363',
            # ax=-1,
            # ay=30,
            bordercolor='rgb(68, 68, 68)',
            borderwidth=3,
            borderpad=8,
            # bgcolor=color(rgb[3], 1.0),
            opacity=1)
        ]

    draw_chart(title, data, format=format, layout=layout)


def draw_dendogram(data, title, labels, format):

    figure = ff.create_dendrogram(data, orientation='left', labels=labels, linkagefun=lambda x: cx.get_linkage(data, 'single', 'euclidean'))

    figure['layout']['title'] = title

    # remove ticks from axis, change font size and remove lines in xaxis
    figure['layout']['xaxis'].update({'ticks': '', 'tickfont': dict(size=24)})
    figure['layout']['yaxis'].update({'ticks': '', 'tickfont': dict(size=24), 'showline': False})


    # margin configuration
    figure['layout']['margin'].update({'b': 40, 'l': 300, 'r': 15, 't': 40, 'pad': 5})

    figure['layout'].update({'autosize': False, 'width': 1200, 'height': 1500})

    fname = title + '_dendogram'

    if format == 'pdf':
        pio.write_image(figure, fname + '.' + format)
    else:
        offline.plot(figure, filename=fname + '.html')



def draw_cf_heatmap(df, title, colorscale, format):
    # df = cf.datagen.heatmap(5,5)
    # print(df.to_iplot())
    figure = df.iplot(kind='heatmap', colorscale=colorscale, title=title, asFigure=True)

    # remove ticks from axis and axis font configuration
    figure['layout']['xaxis'].update({'ticks': '', 'tickfont': dict(size=18)})
    figure['layout']['yaxis'].update({'ticks': '', 'tickfont': dict(size=18)})

    # margin configuration
    figure['layout']['margin'].update({'b': 40, 'l': 120, 'r': 5, 't': 40, 'pad': 5})

    figure['layout'].update({'autosize': False, 'width': 2500, 'height': 2000})

    offline.plot(figure, filename=title + '.html')
    pio.write_image(figure, title + '.' + format)
