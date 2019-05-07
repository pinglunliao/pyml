# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

df_train = pd.read_excel('HourlyWP.xlsx', sheet_name='training dataset')
df_test = pd.read_excel('HourlyWP.xlsx', sheet_name='testing dataset')

def regression(xName, yName):
   # read column radiation from training dataset
    Xtrain = df_train[xName]
    Xtrain = Xtrain.values.reshape(-1,1)
    
    # read column DC_KW from training dataset
    Ytrain = df_train[yName]
    #DC_KW_train = DC_KW_train.values.reshape(-1,1)

    # read column radiation from testing dataset
    Xtest = df_test[xName]
    Xtest = Xtest.values.reshape(-1,1)
    
    # read column DC_KW from testing dataset
    Ytest = df_test[yName]
    #Ytest = Ytest.values.reshape(-1,1)
    
    
    # Linear Regression 
    regr = LinearRegression()
    
    # Train the model using the training sets
    regr.fit(Xtrain, Ytrain)
     
    # Make predictions using the testing set
    Y_pred_lin = regr.predict(Xtest)
    
    mse_lin = mean_squared_error(Ytest, Y_pred_lin)
    mae_lin = mean_absolute_error(Ytest, Y_pred_lin)
    # The mean squared error
    print("Mean squared error: %.2f" % mse_lin)  
    print('Mean absolute error: %.2f' % mae_lin)
    
    # quadratic regression
    poly = PolynomialFeatures(degree = 2)
    
    X_quad_train = poly.fit_transform(Xtrain)
    X_quad_test = poly.fit_transform(Xtest)
    
    quad_regr = LinearRegression()
    quad_regr.fit(X_quad_train, Ytrain)
    
    Y_pred_quad = quad_regr.predict(X_quad_test)
    
    mse_quad = mean_squared_error(Ytest, Y_pred_quad)
    mae_quad = mean_absolute_error(Ytest, Y_pred_quad)
    # The mean squared error
    print("Mean squared error: %.2f" % mse_quad)
    print('Mean absolute error: %.2f' % mae_quad)

    # Plot outputs
    plt.scatter(Xtest, Ytest,  color='black', label='data')
    plt.plot(Xtest, Y_pred_lin, color='red', label='Linear model')
    plt.plot(Xtest, Y_pred_quad, color='blue', label='Quadratic model')
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.title('Regression')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Write result to excel
    writer = pd.ExcelWriter(xName + 'R.xlsx')    
    df = pd.DataFrame( 
            { 'Original': Ytest, 'Linear predicting': Y_pred_lin, 'Quad predicting': Y_pred_quad, 
             'MSE_lin': mse_lin, 'MAE_lin': mae_lin,
             'MSE_quad': mse_quad, 'MAE_quad': mae_quad } )
    df.to_excel(writer, xName + "Pred")
    
    # Save the result 
    writer.save()


regression('radiation', 'DC_KW')
regression('hour', 'DC_KW')