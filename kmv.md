##65783同志merton模型与预测企业违约概率

import numpy as np
from scipy import optimize
from scipy import stats
r=0.0225;sigma_e=0.2893;t=1;E=141276427;DP=1.25e8
def g(w):
    x,sigma_a=w
    N_d1=stats.norm.cdf((np.log(abs(x)*E/DP)+(r+0.5*sigma_a**2)*t)/(sigma_a*np.sqrt(t)))
    N_d2=stats.norm.cdf((np.log(abs(x)*E/DP)+(r-0.5*sigma_a**2)*t)/(sigma_a*np.sqrt(t)))
    #为了防止fsolve迭代到负数报错，给x加绝对值
    e1=E-(x*E*N_d1-DP*N_d2*np.exp(-r*t))
    e2=sigma_e-sigma_a*N_d1*x
    return [e1,e2]#此处返回等于0的式子
result=optimize.fsolve(g,[1,0.1])
result

x,sigma_a=result
DD=(x*E-DP)/(x*E*sigma_a)#违约距离
EDF=stats.norm.cdf(-DD)
print('企业资产为{:.2f},资产的波动率为{:.4%}'.format(x*E,sigma_a))
print('违约距离为DD={:.4f},违约概率EDF={:.4%}'.format(DD,EDF))
