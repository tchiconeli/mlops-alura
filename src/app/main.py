import pandas as pd

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

#comentado ppara substituir por serialização
# arquivo = "/home/chiconeli/Documentos/Aulas/Alura/MLOps/Arquivos/casas.csv"
# #prepara dataset
# df = pd.read_csv(arquivo)
# X = df.drop('preco',axis=1)
# y = df['preco']

# treinamento
# X_train,X_test,y_train,y_test = train_test_split(
#     X, y,test_size=0.3,random_state=42
#     )
# modelo = LinearRegression()
# modelo.fit(X_train,y_train)


#serialização do modelo
modelPath = 'models/modelo.sav'

modelo = pickle.load(open(modelPath,'rb'))
colunas = ['tamanho','ano','garagem']


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME']= os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD']= os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    b = TextBlob(frase)
    tb_en = b.translate(from_lang="pt-br",to='en')
    polaridade=tb_en.sentiment.polarity
    
    return "polaridade: {}".format(polaridade)

# @app.route('/cotacao/<int:tamanho>')
# def cotacao(tamanho):
#     preco = modelo.predict([[tamanho]])
#     return str(preco)

@app.route('/cotacao/',methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json() #recebe o json do metodo POST
    dados_input = [dados[col] for col in colunas] # baseado na variavel colunas, determina a ordem que será lido os dados do json
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True,host='0.0.0.0')
