# from math import nan
#
# import numpy as np
# import pandas as pd
#
# X: pd.DataFrame = pd.DataFrame([
#     [1, 2, 3],
#     [10, 20, 0]
# ])
# # Encode categorical variables
# X = pd.get_dummies(X, drop_first=True)
# print(f"X: ", X)
import sys
import sysconfig

print(sys.exec_prefix)
print(sys.base_exec_prefix)
print(sysconfig.get_config_var("userbase"))
