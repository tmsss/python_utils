import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd


def generate_table(df, max_rows):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows))]
    )


app = dash.Dash()

def generate_layout(title, df, max_rows):
    app.layout = html.Div(children=[
        html.H4(children=title),
        generate_table(df, max_rows)

        app.run_server(debug=True)
    ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
