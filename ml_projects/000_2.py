from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
_reg_coef_ = reg.coef_
ypred = reg.predict([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [2, 2],
    [2, 2],
    [2, 2],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
])

print(f"ypred: ", ypred);
