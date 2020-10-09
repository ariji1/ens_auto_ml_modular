#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from io import BytesIO
import streamlit as st
import pandas as pd
import ppscore as pps
import chardet
from data_clean import clean_data,pred_power

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

@st.cache
def encoder_detector(source_file):
    detected_encoding = chardet.detect(source_file.read())['encoding']
    source_file.seek(0)
    read_data = source_file.read()
    return read_data,detected_encoding

def main():
    st.title("ABC Corp..")
    st.title("Automated Machine Learning Web (POC)")
    #need to look into this for file uploading
    data_file = './DataDump/file' + datetime.now().strftime("%d%b%Y_%H%M%S%f") + '.csv'
    file_bytes = st.file_uploader("Upload a file",encoding=None)
    data_load_state = st.text("Upload your data")
    try:
        if file_bytes is not None:
            #new function test
            main_data,source_encoding = encoder_detector(file_bytes)
            # source_encoding = chardet.detect(file_bytes.read())['encoding']
            target_encoding = "utf-8"
            # file_bytes.seek(0)
            # main_data = file_bytes.read()
            st.write(source_encoding)
            target = open(data_file,"wb")
            target.write(str(main_data, source_encoding).encode(target_encoding))
            dataDF = pd.read_csv(data_file,encoding='utf-8')
            data_load_state.text("Upload done!")
            
    except FileNotFoundError:
        st.error('File not found.')

    st.header("Data Exploration")

    X = ""
    y = ""
    X_train = ''
    X_test = ''
    y_train = ''
    y_test = ''
    y_pred = ''

    # @st.cache
    def load_data():
        df = dataDF.copy()
        return df

    if st.checkbox("Show Data HEAD or TAIL"):
        select_option = st.radio("Select option", ['HEAD', 'TAIL'])
        if select_option == 'HEAD':
            st.write(load_data().head())
        elif select_option == "TAIL":
            st.write(load_data().tail())

    if st.checkbox("Show Full Data"):
        st.write(load_data())
        data_load_state.text("Loading data....Done!")

    if st.checkbox("Data Info"):
        st.text("Data Shape")
        st.write(load_data().shape)
        st.text("Data Columns")
        st.write(load_data().columns)
        st.text("Data Type")
        st.write(load_data().dtypes)
        st.text("Count of NaN values")
        st.write(load_data().isnull().any().sum())
    
    if st.checkbox("Autoclean Data"):
        df = load_data()
        df_new = clean_data(df)
        st.write(df_new.isnull().any())
    
    
    if st.checkbox("Select Target Column"):
        all_columns = load_data().columns
        target = st.selectbox("Select", all_columns)
        st.write(target)
        df_pps,df_pps_top = pred_power(df_new,target)
        st.text("Predictors")
        st.write(df_pps)
        st.text("Recommended predictors for the target")
        st.write(df_pps_top)

    
    # if st.checkbox("Select Target Column"):
    #     all_columns = load_data().columns
    #     target = st.selectbox("Select", all_columns)
    #     st.write(target)
    #     if dataDF[target].dtype == "object":
    #         st.write("You need to convert this target column to numeric value")
    #         label_encoder = LabelEncoder()
    #         dataDF[target] = label_encoder.fit_transform(dataDF[target])
    #         st.write(dataDF[target])
    #     # st.write(load_data()[names])
    # # if st.checkbox("Select to handle null values"):    

    # if st.checkbox("Auto Discard Columns"):
    #     for column in dataDF:
    #         if dataDF[column].nunique() == dataDF.shape[0]:
    #             dataDF.drop([column], axis=1, inplace=True)
    #     for column in dataDF:
    #         if 'name' in column.lower():
    #             dataDF.drop([column], axis=1, inplace=True)

    #     st.text("Data Columns")
    #     st.write(dataDF.columns)
    #     st.text("Count of NaN values")
    #     st.write(dataDF.isnull().any().sum())

    # if st.checkbox("Preprocess Object Type Columns"):
    #     obj_df = dataDF.select_dtypes(include=['object']).copy()
    #     dataDF = dataDF.select_dtypes(exclude=['object'])
    #     try:
    #         one_hot = pd.get_dummies(obj_df)  # ,drop_first=True)
    #     except Exception as e:
    #         print("There has been an exception: ", e)
    #         one_hot = pd.DataFrame()

    #     dataDF = pd.concat([one_hot, dataDF], axis=1)

    # sc = StandardScaler()
    st.header("Split DataSet into Train and Test")

    if st.checkbox("Split"):
        # print(dataDF.dtypes)
        X = df_new[df_pps_top]
        # X = X.apply(normalize)
        y = df_new[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # if st.checkbox("Normalize Columns"):
    #     from sklearn.preprocessing import MinMaxScaler
    #     norm = MinMaxScaler()
    #     X_train = norm.fit_transform(X_train)
    #     X_test = norm.transform(X_test)

    if st.checkbox("Show X_test,X_train,y_test,y_train"):
        st.write("X_train")
        st.write(X_train)
        st.write(X_train.shape)
        st.write("X_test")
        st.write(X_test)
        st.write(X_test.shape)
        st.write("y_train")
        st.write(y_train)
        st.write(y_train.shape)
        st.write("y_test")
        st.write(y_test)
        st.write(y_test.shape)

    def gradBoost(X, y):
        from sklearn.ensemble import GradientBoostingClassifier
        gradientBoosting = GradientBoostingClassifier()
        gradientBoosting.fit(X, y)
        return gradientBoosting

    def randForest(X, y):
        from sklearn.ensemble import RandomForestClassifier
        randomForest = RandomForestClassifier()
        randomForest.fit(X, y)
        return randomForest

    def svm(X, y):
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(X, y)
        return clf

    def xgb(X, y):
        import xgboost as xgboost
        xg_reg = xgboost.XGBRegressor()
        xg_reg.fit(X, y)
        return xg_reg

    def linearReg(X, y):
        from sklearn.linear_model import LinearRegression
        lineReg = LinearRegression()
        lineReg.fit(X, y)
        return lineReg

    def lassoReg(X, y):
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train, y_train)
        return lasso

    if st.checkbox("ML Algorithms"):
        st.write("Available algorithms are:")
        st.write("Binary Classification: GB Classifier, RF Classifier, SVM")
        st.write("Regression: OLS, XGB, Lasso Regression")

        if dataDF[target].nunique() == 2:
            st.header("Using Binary Classification Algorithms")
            GB = gradBoost(X_train, y_train)
            st.write('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(GB.score(X_test, y_test)))
            RF = randForest(X_train, y_train)
            st.write('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, y_test)))
            SVM = svm(X_train, y_train)
            st.write('Accuracy of SVM classifier on test set: {:.2f}'.format(SVM.score(X_test, y_test)))

        elif dataDF[target].nunique() / dataDF[target].count() < .1:
            st.header("Using Multi-Class Classification Algorithms")
            GB = gradBoost(X_train, y_train)
            st.write('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(GB.score(X_test, y_test)))
            st.write(classification_report(y_test, GB.predict(X_test)))
            RF = randForest(X_train, y_train)
            st.write('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, y_test)))
            st.write(classification_report(y_test, RF.predict(X_test)))
        else:
            st.header("Using Regression Algorithms")
            from sklearn.metrics import mean_squared_error, r2_score
            LReg = linearReg(X_train, y_train)
            st.write('R-squared value for Linear Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, LReg.predict(X_test))))
            XGB = xgb(X_train, y_train)
            st.write('R-squared value for eXtreme Gradient Boosting Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, XGB.predict(X_test))))
            LassReg = lassoReg(X_train, y_train)
            st.write('R-squared value for Lasso Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, LassReg.predict(X_test))))


if __name__ == '__main__':
    main()
