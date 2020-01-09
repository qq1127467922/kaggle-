import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#中文乱码处理
plt.rcParams['font.sans-serif'] = 'SimHei' #显示中文不乱码
plt.rcParams['axes.unicode_minus'] = False
#性别向量化
def set_sex(data):
    sex_map = {
        'male':0,
        'female':1
    }
    data['sex'] = data['sex'].map(sex_map)
    return data
#年龄向量化
def set_age(data):
    age_map = {
        '5-14 years':0,
        '15-24 years':1,
        '25-34 years':2,
        '35-54 years':3,
        '55-74 years':4,
        '75+ years':5
    }
    data['age'] = data['age'].map(age_map)
    return data
#世代向量化
def set_generation(data):
    generation_map = {vol:ii for ii,vol in enumerate(set(data['generation']))}
    data['generation'] = data['generation'].map(generation_map)
    return data

def set_gdp_str_to_float(data):
    def gdf_to_float(gdp_string):
        return float(gdp_string.replace(',',''))
    data[' gdp_for_year ($) '] = data[' gdp_for_year ($) '].apply(gdf_to_float)
    return data
def pop_10w_to_1000w(number):
    return int(number*100)
#相关性分析
def get_correlation_coefficient(data):
    cols = ['sex', 'age', 'suicides_no', 'population', 'suicides/100k pop',' gdp_for_year ($) ','gdp_per_capita ($)','generation']
    cm = np.corrcoef(data[cols].values.T)#corrcoef方法按行计算皮尔逊相关系数,cm是对称矩阵
    sns.set(font_scale=0.8) #font_scale设置字体大小
    sns.set(style='whitegrid', context='notebook')
    hm = sns.heatmap(cm, cbar=True, annot=True,
                     cbar_ax=None,ax=None,
                     square=False,
                     fmt='.2f',
                     linewidths=0.05,
                     annot_kws={'size': 12},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.show()
#统计31年各国家自杀人数
def get_die_sum_for_country(data):
    country_list = set(data['country'])
    people_sum = []
    for country_ in country_list:
        people_sum.append(sum(data[data['country']==country_]['suicides_no'].values))
    country = country_list
    for i in range(len(country_list)):
        if people_sum[i]>50000:
            print('国家:'+list(country_list)[i] + '自杀人数:'+str(people_sum[i]))
    plt.title('Count of suicides for 31 years',fontdict={'size' : 25})
    #sns.set(font_scale=1.5)
    plt.pie(people_sum)
    plt.tight_layout()
    plt.show()
#统计每年自杀人数
def get_die_sum_for_year(data):
    suic_sum_yr = pd.DataFrame(data['suicides_no'].groupby(data['year']).sum())
    suic_sum_yr = suic_sum_yr.reset_index().sort_index(by='suicides_no', ascending=False)
    most_cont_yr = suic_sum_yr
    fig = plt.figure(figsize=(30, 10))
    plt.title('Count of suicides for years',fontdict={'size' : 25})
    sns.set(font_scale=2.5)
    sns.barplot(y='suicides_no', x='year', data=most_cont_yr, palette="OrRd");
    plt.ylabel('Count of suicides',fontdict={'size' : 25})
    plt.xlabel('',fontdict={'size' : 25})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xticks(rotation=270)
    plt.tight_layout()
    plt.show()
#训练knn模型
def train_knn_model(train_x,train_y):
    knn_model = KNeighborsClassifier()
    knn_model.fit(train_x, train_y)
    return knn_model
#训练svm模型
def train_svm_model(train_x,train_y):
    svm_model = SVC()
    svm_model.fit(train_x,train_y)
    return svm_model
#训练提升树模型
def train_gradient_model(train_x,train_y):
    gradient_model = GradientBoostingRegressor()
    gradient_model.fit(train_x,train_y)
    return gradient_model
#模型评估
def train_test(data):
    feature = ['sex','age','suicides_no','population',' gdp_for_year ($) ','gdp_per_capita ($)']
    x_data = data[feature].astype(float)
    y_data = data['suicides/100k pop']
    y_data = y_data.apply(pop_10w_to_1000w)
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.01, random_state=50)
    knn_model = train_knn_model(X_train,y_train)
    svm_model = train_svm_model(X_train,y_train)
    gradient_model = train_gradient_model(X_train,y_train)
    knn_pred = knn_model.predict(X_test)
    knn_score = knn_model.score(X_test,y_test)
    knn_loss = explained_variance_score(y_test,knn_pred)
    svm_pred = svm_model.predict(X_test)
    svm_score = svm_model.score(X_test,y_test)
    svm_loss = explained_variance_score(y_test,svm_pred)
    gradient_pred = gradient_model.predict(X_test)
    gradient_score = gradient_model.score(X_test,y_test)
    gradient_loss = explained_variance_score(y_test,gradient_pred)
    # print('knn模型loss为:'+str(knn_loss))
    print('knn模型准确率:'+str(knn_score))
    # print('svm模型loss为:'+str(svm_loss))
    print('svm模型准确率:'+str(svm_score))
    # print('提升树模型loss为:'+str(gradient_loss))
    print('提升树模型准确率为:'+str(gradient_score))
    x_label = np.linspace(1,len(y_test),len(y_test))
    plt.plot(x_label,y_test)
    plt.plot(x_label,knn_pred)
    plt.plot(x_label,svm_pred)
    plt.plot(x_label,gradient_pred)
    plt.legend(['真实值','knn预测值','svm预测值','提升树预测值'])
    plt.show()
if __name__ == '__main__':
    data = pd.read_csv('master.csv')
    # 查看含有缺失值的列
    print(data.isnull().any())
    data = data.dropna()
    print(data.isnull().any())
    # print(set_sex(data)['sex'])
    # print(set_age(data)['age'])
    #print(set_generation(data)['generation'])
    #print(data[' gdp_for_year ($) '])
    #print(set_gdp_str_to_float(data)[' gdp_for_year ($) '])
    data = set_gdp_str_to_float(set_generation(set_age(set_sex(data))))
    #print(data.info())
    print(data[['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']].describe())
    #get_correlation_coefficient(data)
    #get_die_sum_for_country(data)
    #get_die_sum_for_year(data)
    train_test(data)
