import xlrd 
import numpy as np
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import pickle as pkl


def get_model(material,worksheet):
    mats = worksheet.col_values(0)
    the_row = mats.index(material)
    x_raw = worksheet.row_values(1)[2:15]
    # x = np.array(x)
    y_raw = worksheet.row_values(the_row)[2:15]
    x = []
    y = []
    for i in range(len(x_raw)):
        if y_raw[i] !='':
            x.append(x_raw[i])
            y.append(y_raw[i])
    x = 1/np.array(x)
    x = x**2
    x = x.reshape(-1,1)
    y = np.array(y) 
    mod = pl.make_pipeline(
        sp.PolynomialFeatures(3),
        lm.LinearRegression()
    )
    mod.fit(x,y)
    modelname = 'models/'+material+'.pkl'
    with open(modelname, 'wb') as f:
        pkl.dump(mod, f)
    
def get_glass_mod_lib(filename):
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    mats = worksheet.col_values(0)[1:-1]
    for each in mats:
        if each != '' or each.strip()[:5] != 'Over!':
            get_model(each,worksheet)


if __name__ == '__main__':
    filename = 'o_glasslib.xls'
    get_glass_mod_lib(filename)

    # x = (1/np.linspace(706,350,365))**2
    # x = x.reshape(-1,1)

    # m = get_model('F1')
    # pred_y = m.predict(x)
    # print(pred_y)
    


    