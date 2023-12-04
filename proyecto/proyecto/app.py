from flask import Flask, render_template, request
import pandas as pd
from process import Process


app = Flask(__name__)

mi_objeto = Process()

@app.route('/')
def initial_form():
    # Cargar el modelo desde el archivo
    return render_template('index.html', form=[])

@app.route('/multiple')
def multiple_form():
    # Cargar el modelo desde el archivo
    return render_template('multiple.html')

@app.route('/multiple/submit', methods=['POST'])
def submit_multiple_form():
    result, df = mi_objeto.get_model_result_multiple(request)
    mi_objeto.get_torta(result)
    mi_objeto.get_distribution()
    mi_objeto.get_multiple_graphic(df)
    return render_template('multiple.html', 
                           is_calculated=True,
                           down_media=result['POR_ENCIMA_MEDIA'].isin([True]).sum(),
                           up_media=result['POR_ENCIMA_MEDIA'].isin([False]).sum(),
                           size=len(result)
                           )

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        result, form = mi_objeto.get_model_result_one(request)
        return render_template('index.html', result=result, form= form, is_calculated=True)

if __name__ == '__main__':
    app.run(debug=True)
