#Análise exploratória dos dados
#Importando as bibliotecas
import os
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import joblib

#####EXTRACTION
KAGGLE_DATASET = "kaverappa/amazon-best-seller-softwares"
DATASETS_PATH = kagglehub.dataset_download("kaverappa/amazon-best-seller-softwares")


#####LOAD
def load_abss_data(dataset_path=DATASETS_PATH, filename="best_sellers_data2.csv"):
    csv_path = os.path.join(dataset_path, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"arquivo nao encontrado no path")
    return pd.read_csv(csv_path)

df=load_abss_data()

#tratamento de dados
## Verificando os dados
print(df.head())
print('\n')
print(df.describe())
print('\n')
print(df.info())

#Limpeza de dados incluindo: transformação dos preços em numeric, remoção de duplicatas e de valores nulos
def clean_data(df):
    df['product_price']= df['product_price'].replace({'\$': '', ',':''}, regex=True)
    df['product_price'] = pd.to_numeric(df['product_price'], errors='coerce')
    print('\n','Null Count:',df.isnull().sum())
    df.dropna(subset=['product_price', 'product_star_rating', 'product_num_ratings'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['country'] = df['country'].astype('category')
    
    return df
#Garantindo que os dados estão limpos e prontos para uso
df= clean_data(df)
print('\nPos-limpeza info dos dados:')
print(df.info())

# Análise exploratória dos dados
#1. Count plot: Distribuição dos produtos por pais
c_plot = sns.countplot(data= df, x='country',hue='country' ,order=df['country'].value_counts().index)
plt.title('Distribuição de produtos nos países')
plt.xlabel('Países')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

###Nesse gráfico foi gerado apenas para entender de onde a maior parte dos produtos vem, nesse caso, dos EUA.

#2. Histograma: Distribuição dos produtos por nota
h_plot = sns.histplot(data=df, x='product_star_rating',bins=50, kde=True)
plt.title('Distribuição de produtos por nota')
plt.xlabel('Nota')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()
###Nesse gráfico foi gerado para entender a distribuição das notas dos produtos, onde a maioria dos produtos tem notas entre 4 e 5 estrelas.

#3. Box plot: Distribuição dos preços para identificar outliers
b_plot = sns.boxplot(data=df, x='product_price')
plt.title('Distribuição de preços')
plt.xlabel('Preços dos produtos')
plt.tight_layout()
plt.show()

p99 = df['product_price'].quantile(0.99)
df=df[df['product_price']<=p99]

#3.1 Box plot: Distribuição dos preços após remoção de outliers
b_plot = sns.boxplot(data=df, x='product_price')
plt.title('Distribuição de preços após remoção de outliers')
plt.xlabel('Preços dos produtos')
plt.tight_layout()
plt.show() 
# Esse gráfico demonstra que a maioria dos produtos (representados pela barra azul) estão entre 0 e 200 dolares. 
# No entanto, é possível notar uma quantidade de grande de outliers até 6000 dólares que não devem ser ignorados, por isso, a técnica de limpeza desses dados foi a de percentile 99. 
# Dessa forma, garantindo que seriam eliminados apenas os produtos que realmente tivessem uma quantiade pequena. 


#4. Pair plot: Relações entre variaveis numéricas
p_plot = sns.pairplot(df[['product_price', 'product_star_rating', 'product_num_ratings', 'rank']])
plt.title('Relações entre variáveis numéricas')
plt.show()
# Esse gráfico mostra a relação entre as variáveis numéricas, assim sendo possível notar que a maioria dos produtos com preço mais baixo tem uma quantidade maior de estrelas e de avaliações.
# Assim, é possível notar que os produtos mais caros não necessariamente vistos como melhores pelo público, já que, apesar de terem otimas avaliações, os produtos mais baratos também são bem avaliados.

#5. Heatmap de correlação: Apenas se tiver +4 colunas numéricas
numeric_df= df.select_dtypes(include=[np.number])
if numeric_df.shape[1]>=4:
    plt.figure(figsize=(8,6))
    corr = numeric_df.corr()
    sns.heatmap(data=corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap de correlação')
    plt.tight_layout()
    plt.show()
else:
    print("Sem colunas numericas suficientes")
# Nesse gráfico é possível notar que a variável 'product_star_rating' tem uma correlação positiva com 'product_num_ratings' e 'rank', o que indica que produtos com mais avaliações tendem a ter uma classificação mais alta.
# Além disso, a variável 'product_price' tem uma correlação negativa com 'product_star_rating', o que sugere que produtos mais caros não necessariamente são melhor avaliados.
# Assim, sendo um indicativo que o preço não é o único fator que influencia a avaliação dos produtos, sendo possível que características como qualidade, marca e funcionalidade também desempenhem um papel importante na percepção dos consumidores.
# Por fim, as variáveis não apresentam uma correlação muito forte entre si, o que sugere que cada uma delas traz informações distintas sobre os produtos.


#Testes de modelos de predição
#Dividindo as features, target e o dataset em treino e teste
#Features e target
features = ['product_star_rating', 'product_num_ratings', 'rank']
target = 'product_price'

X= df[features]
y= df[target]

#Treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=RANDOM_STATE)

## Modelo de regressão linear
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
price_pred1= lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, price_pred1)
lin_rmse = np.sqrt(lin_mse)
print('\nErro quadrático médio:', lin_rmse) 
# O erro quadrático médio é uma métrica para entender o desempenho do modelo, onde quanto menor o valor, melhor o modelo.
# Nesse caso, o modelo de regressão linear teve um erro quadrático médio de $974 dólares, indicando que ele não é um bom modelo para prever o preço dos produtos, já que 
# o preço dos produtos varia entre 0 e 6000 dólares.

#Cross-validation: Regressão linear
lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
lin_rsme_scores = np.sqrt(-lin_scores)
print('\n', 'RMSE lin reg:',lin_rsme_scores)
lin_r2 = r2_score(y_test, price_pred1)
print('\n', 'R2 lin reg:',lin_r2)
# Como era de se esperar pelo RMSE apresentado anteriormente, o R2 também não é bom, cerca de 4,42% ,já que o modelo não consegue explicar a variabilidade dos dados.

#Scatter plot: prediçoes vs. preços reais - Regressão linear
plt.scatter(x=y_test, y=price_pred1, alpha=0.3)
plt.title('Prediçao dos preços vs. Preços reais - Regressão Linear')
plt.xlabel('Preços reais')
plt.ylabel('Predição de preços')
plt.plot([y.min(), y.max()], [y.min(), y.max()],color='red', lw=2)
plt.tight_layout()
plt.show()
# O gráfico nos mostra visualmente que os preços reais e as predições do modelo de regressão linear estão muito distantes do que se deseja. 

## Modelo de decision tree
tree_reg= DecisionTreeRegressor(random_state=RANDOM_STATE)
tree_reg.fit(X_train, y_train)
price_pred2 = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, price_pred2)
tree_rmse= np.sqrt(tree_mse)
print('\nErro quadrático médio:', tree_rmse) 
# O erro quadrático médio já é consideravelmente menor do que o apresentado pela regressão linear, sendo um modelo um pouco melhor.
# Dessa vez, sendo apresentada uma diferença média de $505 dolares entre os preços reais e os preços preditos.

#Cross-validation: Decision Tree Regressor
tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
print('\n', 'RMSE tree reg:',tree_rmse_scores)
tree_r2 = r2_score(y_test, price_pred2)
print('\n', 'R2 tree reg:',tree_r2)
# Novamente, o R2 do modelo de árvore de decisão é um pouco melhor do que do modelo anterior, sendo cerca de 74,25%. O que não é o melhor resultado, mas já é uma melhora considerável.

#Scatter plot: predições de preços vs. preços reais - Regressão de Árvore de decisão
plt.scatter(x=y_test, y=price_pred2, alpha=0.3)
plt.title('Prediçao dos preços vs. Preços reais - Decision Tree Regressor')
plt.xlabel('Preços reais')
plt.ylabel('Preços preditos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color= 'red', lw=2)
plt.tight_layout()
plt.show()
# O gráfico mostra que a Decision Tree Regressor consegue predizer alguns valores no eixo esperado, mas ainda há algumas predições fora do eixo.

## Modelo de random forest regressor
rdmf_reg = RandomForestRegressor(random_state=RANDOM_STATE)
rdmf_reg.fit(X_train, y_train)
price_pred3 = rdmf_reg.predict(X_test)
rdmf_mse = mean_squared_error(y_test, price_pred3)
rdmf_rmse = np.sqrt(rdmf_mse)
print('\nErro quadrático médio:', rdmf_rmse) 
# O erro quadrático médio é de $546 dolares.
# É um erro um pouco maior do que o da decision tree mas ainda bem menor do que o da regressão linear, sendo um modelo que fica entre o meio termo, mais pro lado da Decision Tree.

#Cross-validation: Random Forest Regressor
rdmf_scores = cross_val_score(rdmf_reg,X_train, y_train, scoring='neg_mean_squared_error', cv=10)
rdmf_rmse_scores= np.sqrt(-rdmf_scores)
print('\n', 'RMSE random forest reg:',rdmf_rmse_scores)
rdmf_r2 = r2_score(y_test, price_pred3)
print('\n', 'R2 random forest reg:',rdmf_r2)
# O R2 do modelo de Random Forest Regressor é de 69,97%, o que é um resultado um pouco pior do que o da Decision Tree, mas ainda assim, um resultado bom.

#Scatter plot: predições dos preçps vs. preços reais - Random Forest Regressor
plt.scatter(x=y_test, y=price_pred3, alpha=0.3)
plt.title("Predição dos preços vs. Preços reais - Random Forest Regressor")
plt.xlabel('Preços reais')
plt.ylabel('Predição dos preços')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.tight_layout()
plt.show()
# O gráfico mostra que o modelo de Random Forest Regressor preve alguns valores no eixo esperado, mas tem uma maior dificuldade em predizer os produtos que de fato teriam um valor maior.

#Aperfeiçoamento do modelo de Decision Tree Regressor
#GridSearchCV
param_grid= {'max_depth': [3, 5,8,10, 12, None],
     'min_samples_split': [2, 5,10],
     'min_samples_leaf': [1,2,4],
     'max_features': [None, 'sqrt', 'log2']
     }
cv= KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(estimator=tree_reg, param_grid= param_grid, cv=cv,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,
                           return_train_score=True)
grid_search.fit(X_train, y_train)

print('\n','Melhores hiperparametros encontrados:',grid_search.best_params_)
print('\n','Melhor modelo treinado:',grid_search.best_estimator_)

final_model_gs = grid_search.best_estimator_
final_model_gs.fit(X_train, y_train)
final_model_gs_price_pred = final_model_gs.predict(X_test)
final_model_gs_mse = mean_squared_error(y_test, final_model_gs_price_pred)
final_model_gs_rmse = np.sqrt(final_model_gs_mse)
print('\n', 'Final model GS RMSE:',final_model_gs_rmse)
final_model_gs_r2 = r2_score(y_test, final_model_gs_price_pred)
print('\n', 'Final model GS R2:',final_model_gs_r2)
# O modelo de Decision Tree Regressor com GridSearchCV teve um erro quadrático médio de $526 dólares e um R2 de 72%, indicando que o modelo treinado sem o GridSearch operou melhor
# do que o modelo treinado com o GridSearch, mesmo com os melhores hiperparâmetros encontrados, evidenciando que os melhores hiperparametros não necessariamente são os que vão gerar o melhor modelo.

#RandomSearchCV
param_rdm_search = {'max_depth': [3, 5,8,10, 12, None],
     'min_samples_split': [2, 5,10],
     'min_samples_leaf': [1,2,4],
     'max_features': [None, 'sqrt', 'log2']
     }

random_search = RandomizedSearchCV(tree_reg, param_distributions=param_rdm_search,
                                   n_iter=10,
                                   cv=cv,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   return_train_score=True,
                                   random_state= RANDOM_STATE)
random_search.fit(X_train, y_train)
print('\n','Melhores hiperparametros encontrados:',random_search.best_params_)
print('\n','Melhor modelo treinado:',random_search.best_estimator_)
final_model_rdm = random_search.best_estimator_
final_model_rdm.fit(X_train, y_train)
final_model_rdm_price_pred = final_model_rdm.predict(X_test)
final_model_rdm_mse = mean_squared_error(y_test, final_model_rdm_price_pred)
final_model_rdm_rmse = np.sqrt(final_model_rdm_mse)
print('\n', 'Final model RDM RMSE:',final_model_rdm_rmse)
final_model_rdm_r2 = r2_score(y_test, final_model_rdm_price_pred)
print('\n', 'Final model RDM R2:',final_model_rdm_r2)

# O refinamento com Random SearchCV teve um erro quadrático médio de $773 dólares e um R2 de 39,80%. Assim, indicando que o modelo treiando com o RandomSearch não foi bom.
# O que mostra que o modelo de Decision Tree Regressor treinado sem nenhum aperfeiçoamento foi o melhor modelo entre os três modelos testados e mesmo comparado com o próprio aperefeiçoado.

#Comparação dos resultados obtidos com GSearch e RSearch
#Comparação do tempo de execução
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()
print(f"Tempo de execução do GridSearchCV: {end - start:.2f} segundos")

start = time.time()
random_search.fit(X_train, y_train)
end = time.time()
print(f"Tempo de execução do RandomizedSearchCV: {end - start:.2f} segundos")

#O tempo de execução do Random Search foi bem mais rápido do que o tempo do GridSearch, no entanto foi possível perceber que o ganho com um tempo menor muito provavelmente não seria um ponto de decisão para usar o modelo identificado pelo RSearch,
# já que o modelo do GSearch teve parametros melhores entre os dois.

#Visualização dos resultados RMSE
plt.figure(figsize=(8,5))
plt.bar(['GridSearchCV', 'RandomSearchCV'], [final_model_gs_rmse, final_model_rdm_rmse],color = ['blue', 'orange'])
plt.title('Comparação dos RMSE dos modelos')
plt.ylabel('RMSE')
plt.xlabel('Modelos')
plt.tight_layout() 
plt.show()

#Comparação dos scores de treino e validação	
mean_train_score_gs = np.mean(grid_search.cv_results_['mean_train_score'])
mean_val_score_gs = np.mean(grid_search.cv_results_['mean_test_score'])
mean_train_score_rdm = np.mean(random_search.cv_results_['mean_train_score'])
mean_val_score_rdm = np.mean(random_search.cv_results_['mean_test_score'])

print('\n','GridSearchCV - Média de treino:',mean_train_score_gs)
print('\n','GridSearchCV - Média de validação:',mean_val_score_gs)
print('\n','RandomSearchCV - Média de treino:',mean_train_score_rdm)
print('\n','RandomSearchCV - Média de validação:',mean_val_score_rdm)

#As médias de avaliação são apresentadas em neg_mean_squared_error, onde quanto menor o valor, melhor o modelo.
# Dessa forma, o modelo de GridSearch teve médias piores do que o modelo de RandomSearch, apesar de apresentar um melhor modelo no final.
# Isso pode signficar que o modelo de GridSearch, por testar todos os modelos possíveis, a média foi puxada para baixo, enquanto o modelo de RandomSearch, por testar menos modelos, teve uma média melhor.

#Salvando os modelos com joblib
#modelo regressão linear 
joblib.dump(lin_reg, 'lin_reg_model.pkl')
#modelo decision tree   
joblib.dump(tree_reg, 'tree_reg_model.pkl')
#modelo random forest
joblib.dump(rdmf_reg, 'rdmf_reg_model.pkl')
#modelo decision tree + grid search
joblib.dump(final_model_gs, 'final_model_gs.pkl')
#modelo decision tree + random search
joblib.dump(final_model_rdm, 'final_model_rdm.pkl')
