"""DashApp used for visualization of model fits.

   Runs the app.

"""

import dash
import dashapp.callbacks.callbacks as callbacks
import dashapp.layout.layout as layout

# Load external style sheets.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize app.
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Setup app.
app.layout = layout.create_layout()
callbacks.register_callbacks(app)

# Run app.
if __name__ == '__main__':
    app.run_server(debug=True)
