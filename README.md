# Credit Fraud Detection
## Descrição
Este projeto visa detectar fraudes em transações de crédito utilizando diversas técnicas de aprendizado de máquina. O conjunto de dados é fornecido em um arquivo ZIP e contém informações sobre transações de cartão de crédito. O objetivo é identificar transações fraudulentas, que são muito menos frequentes do que as transações legítimas.

## Requisitos
 - Python 3.x
 - Google Colab (opcional, para montar o Google Drive)
 - Bibliotecas: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, pyyaml
 - Estrutura do Projeto
 - Imports e Configuração Inicial

## Montagem do Google Drive (opcional)

 - Importação de bibliotecas necessárias
 - Configuração do semente aleatória

## Importação dos Dados

 - Carregamento e visualização inicial dos dados
 - Balanceamento dos dados para garantir uma proporção adequada entre transações fraudulentas e não fraudulentas

## Visualização dos Dados
 - Aplicação de técnicas de redução de dimensionalidade (t-SNE e TruncatedSVD)
 - Visualização dos dados balanceados
 - 
## Modelos de Detecção de Fraude

 - Logistic Regression: Modelo de regressão logística para classificação
 - Isolation Forest: Algoritmo de detecção de anomalias baseado em árvores
 - One-Class SVM: Máquina de vetores de suporte para detecção de anomalias
 - Local Outlier Factor (LOF): Fator de Anomalia Local
 - DBSCAN: Algoritmo de clustering baseado em densidade para detecção de anomalias
 - XGBoost: Classificador XGBoost para detecção de fraudes

## Avaliação dos Modelos

 - Comparação das previsões dos modelos com os dados reais
 - Exibição de relatórios de classificação e matrizes de confusão

## Exemplos

### Importação dos Dados

```python
from zipfile import ZipFile
zip_file = ZipFile('/content/drive/MyDrive/CreditFraud.zip')
data = pd.read_csv(zip_file.open('creditcard.csv'))
```
### Treinamento e Avaliação do Modelo de Regressão Logística

```python
from sklearn.linear_model import LogisticRegression
regressorL = LogisticRegression(max_iter=1000)
regressorL.fit(X_train, y_train)
y_regressor = regressorL.predict(X_test)
```

### Visualização dos Resultados

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, y_regressor))
ConfusionMatrixDisplay.from_predictions(y_test, y_regressor)
plt.show()
```

Licença
Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

