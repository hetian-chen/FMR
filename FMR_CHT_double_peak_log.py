import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import os,sys

os.chdir(sys.path[0]) #使得在vscode中也可以使用相对路径

###########################################################
####             需要在这里修改参数                    #####
###########################################################

### 文件名字：
file_name = 'input.log'
### 需要拟合的频率
freq_list = [6.0,8.0,10.0,12.0,14.0]
### 是否有峰重叠
peak_n = 2  #如果没有洛伦兹峰重合设位1，有两个洛伦兹峰重合设位2

class Ansys():
    def __init__(self,name='BFO_LSMO_1208P1N1 - 20210114-111847.log',header=36,fFMR = 6.0,peak_num = 2,H_range=2000):
        self.name = name                                                 #文件名
        self.header = header                                             #表头长度
        self.x = 'Field'
        self.y = 'IQ'
        self.frequency = fFMR                                            #选取的频率
        self.peak_num = peak_num
        self.x_range = [-np.inf,np.inf]             
        self.H_range = H_range                                           #要拟合的磁场范围
        self.df = self.read(self.name)                                   #在此步自动调用读取
        if peak_num == 1:
            self.returndata = {'fFMR':fFMR,'Ks':0,'Kas':0,'delta_H':0,'Hres':0,'C':0}
        elif peak_num == 2:
            self.returndata = {'f_FMR':fFMR,'K_s':0,'K_as':0,'delta_H':0,'H_res':0,'K_s2':0,'K_as2':0,'delta_H2':0,'H_res2':0,'C':0}
        self.y_pre =   np.array([0])
        self.single1 = np.array([0])
        self.single2 = np.array([0])

    def read(self,name):  ##读取文件
        self.name = name
        df = pd.read_csv('./'+name,sep='\t',header = self.header)  ##去除表头
        # print(df)                                                  #用于测试
        ##提取所需的行
        df2 = df[df['Frequency']==self.frequency]
        df3 = df2[df2[self.x] > self.x_range[0]]
        df4 = df3[df3[self.x] < self.x_range[1]]
        return df4

    def new_fit(self,df):
        def lorentzian_derivative_guess(df):
            """
            Guessing for parameters in the lorentzian derivative, for use with
            fit_dataset
            """
            H_peak_h = df[self.x][df[self.y].idxmax()]
            H_peak_l = df[self.x][df[self.y].idxmin()]

            resonance_guess = (H_peak_h+H_peak_l)/2
            peak_to_peak = np.abs(H_peak_l-H_peak_h)
            linewidth_guess =  peak_to_peak

            if H_peak_h < H_peak_l:
                amplitude_guess = -(df[self.y].max() - df[self.y].min())*linewidth_guess
            else:
                amplitude_guess = (df[self.y].max() - df[self.y].min())*linewidth_guess

            asym_amplitude_guess = amplitude_guess


            self.x_range[0] = resonance_guess - self.H_range/2           #设置要拟合的范围
            self.x_range[1] = resonance_guess + self.H_range/2

            self.df = self.df[self.df[self.x] > self.x_range[0]]
            self.df = self.df[self.df[self.x] < self.x_range[1]]
            
            if self.peak_num == 1:
                return np.array([amplitude_guess, resonance_guess, linewidth_guess, asym_amplitude_guess, 0.0])
            elif self.peak_num == 2: 
                return np.array([amplitude_guess, resonance_guess, linewidth_guess, asym_amplitude_guess,amplitude_guess, 0.8*resonance_guess, linewidth_guess, asym_amplitude_guess, 0.0])


        def lorentzian_derivative_sym(H, Ka, Hres, delta_H,Kas, C):
            '''
            洛伦兹函数
            '''
            H_minus = H-Hres
            half_delta_H = delta_H/2
            return Ka*half_delta_H*H_minus/                                    ((half_delta_H**2.0*(1.0 + H_minus**2.0/half_delta_H**2.0))**2.0) \
                + Kas*(half_delta_H**2.0*(1.0-H_minus**2.0/half_delta_H**2.0))/((half_delta_H**2.0*(1.0 + H_minus**2.0/half_delta_H**2.0))**2.0) + C

        def lorentzian_derivative_sym_2(H, Ka, Hres, delta_H,Kas, Ka2, Hres2, delta_H2,Kas2, C):    #双重峰的洛伦兹函数
            return lorentzian_derivative_sym(H, Ka, Hres, delta_H,Kas,0) + lorentzian_derivative_sym(H, Ka2, Hres2, delta_H2,Kas2,C) 


        if self.peak_num == 1:
            p01 = list(lorentzian_derivative_guess(df))
            print('---------------------------------------')
            print('gauss value is',p01)
            para =   optimize.curve_fit(lorentzian_derivative_sym, self.df[self.x], self.df[self.y],p0 = p01,method = 'trf',bounds = ((-np.inf,0,0,-np.inf,-1),(np.inf,np.inf,np.inf,np.inf,1)), maxfev=10000)[0]
            y_pre = np.array([lorentzian_derivative_sym(H, para[0],para[1],para[2],para[3],para[4])  for H in self.df[self.x] ])
            print('frequency {} is done'.format(self.frequency))
            print('fitting parameter is',para)
            for i,name in enumerate(['K_s','H_res','delta_H','K_as','C']):
                self.returndata[name] = para[i]
            single1 = 'None'
            single2 = 'None'
            return y_pre,single1,single2

        elif self.peak_num == 2:
            p02 = list(lorentzian_derivative_guess(df))
            print('--------------------------------------------')
            print('gauss value is',p02)
            para =  optimize.curve_fit(lorentzian_derivative_sym_2, self.df[self.x], self.df[self.y], p0 =p02,method = 'trf',bounds = ((-np.inf,0,0,-np.inf,-np.inf,0,0,-np.inf,-1),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1)), maxfev=10000)[0]
            print('frequency {} is done'.format(self.frequency))
            if para[5] > para[1]:  #比较H_res和H_res2，使得两个峰在不同频率下的对应
                for i in range(4):
                    para[i],para[i+4] = para[i+4],para[i]
            print('fitting parameter is',para)
            y_pre = np.array([lorentzian_derivative_sym_2(H, para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7],para[8])  for H in self.df[self.x] ])
            single1 = np.array([lorentzian_derivative_sym(H, para[0],para[1],para[2],para[3],para[8])  for H in self.df[self.x] ])
            single2 = np.array([lorentzian_derivative_sym(H, para[4],para[5],para[6],para[7],para[8])  for H in self.df[self.x] ])
            for i,name in enumerate(['K_s','H_res','delta_H','K_as','K_s2','H_res2','delta_H2','K_as2','C']):
                self.returndata[name]      = para[i]
            return y_pre,single1,single2

    def plot_out(self):
        self.y_pre,self.single1,self.single2 = self.new_fit(self.df)

class Fit():
    def __init__(self,H_res_n='H_res',delta_H_n='delta_H'):
        self.gamma = 0
        self.H_k = 0
        self.M_s =0
        self.damping = 0
        self.delta_H0 = 0
        self.H_res_n = H_res_n
        self.delta_H_n = delta_H_n

    @staticmethod
    def change_unit(x,type):
        if type == 'field':
            return x/(4*np.pi)*1E3
        elif type == 'frequency':
            return 1E9*x

    def fit_Kittel_functions(self,df,mode='in_plane'):
        def Kittel_funtions(H_res,gamma,M_s,H_k):
            miu_0 = 4*np.pi*1E-7
            if mode == 'in_plane' or 'inplane' or 'in' or 'ip':
                return (gamma*miu_0/(2*np.pi)) * np.sqrt((M_s+H_k+np.abs(H_res))*(H_k+np.abs(H_res)))
            elif mode == 'out_of_plane' or 'out_plane' or 'out' or 'out_plane' or 'oop':
                return (gamma*miu_0/(2*np.pi)) * (M_s+H_k+np.abs(H_res))

        gamma,M_s,H_k = optimize.curve_fit(Kittel_funtions, df[self.H_res_n], df['f_FMR'],p0 = [10000,10000,10000], bounds = ((0,0,0),(np.inf,np.inf,np.inf)),method = 'trf',maxfev=8000)[0]
        self.gamma = gamma
        f_pre = [Kittel_funtions(H_res,gamma,M_s,H_k) for H_res in df[self.H_res_n]]
        print('--------------------------------------------------------------')
        print('M_s', M_s, 'A/m','   miu0 * M_s=', 4*np.pi*1E-7*M_s, 'T')
        print('H_k =',H_k,'A/m,   =',H_k*4*np.pi*1E-3,'Oe')
        print('gamma/2Pi = ',gamma/(2*np.pi)*1E-9,'GHz T-1')
        self.H_k = H_k
        self.M_s = M_s
        return f_pre


    def fit_damping_function(self,df):
        def damping_function(f,delta_H0,alpha):
            miu_0 = 4*np.pi*1E-7
            return delta_H0 + (4*np.pi*alpha*f/self.gamma)/miu_0

        delta_H0, alpha = optimize.curve_fit(damping_function, df['f_FMR'], np.abs(df[self.delta_H_n]),method = 'trf',maxfev=8000)[0]
        H_pre = [damping_function(f,delta_H0,alpha) for f in df['f_FMR']]
        print('--------------------------------------------------------------')
        print('delta_H0 =',delta_H0, 'A/m,   =',delta_H0*4*np.pi*1E-3,'Oe')
        print('alpha = ',alpha)
        self.damping = alpha
        self.delta_H0 = delta_H0
        return H_pre

    
    
if __name__ == '__main__':  
    fig = plt.figure(figsize=(12,6))
    ax = [] 

    with open(file_name) as f:
        lines = f.readlines()
        header_n = lines.index('[Data]\n')-1

    if len(freq_list) == 1:
        ax_line = 1
        ax_row = 3
    elif len(freq_list) == 2:
        ax_line = 2
        ax_row = 2
    elif len(freq_list) == 3 or len(freq_list) == 4:
        ax_line = 2
        ax_row = 3
    elif len(freq_list) == 5 or len(freq_list) == 6:
        ax_line = 2
        ax_row = 4
    elif len(freq_list) == 7:
        ax_line = 3
        ax_row = 3
    elif len(freq_list) == 8 or len(freq_list) == 9 or len(freq_list) == 10:
        ax_line = 3
        ax_row = 4
    elif len(freq_list) == 11 or len(freq_list) == 12 or len(freq_list) == 13 or len(freq_list) == 14:
        ax_line = 4
        ax_row = 4
    else:
        print('too much frequency is selected, please enter less frequency')

    if peak_n == 1:
        returndata = {'f_FMR':[],'K_s':[],'K_as':[],'delta_H':[],'H_res':[],'C':[]}
    elif peak_n == 2:
        returndata = {'f_FMR':[],'K_s':[],'K_as':[],'delta_H':[],'H_res':[],'K_s2':[],'K_as2':[],'delta_H2':[],'H_res2':[],'C':[]}

    for i,f in enumerate(freq_list):  #绘制文件一
        ax.append(fig.add_subplot(ax_line,ax_row,i+1))
        a = Ansys(name=file_name,header=header_n,fFMR = f,peak_num = peak_n,H_range=2000)
        a.plot_out()
        ax[i].plot(a.df[a.x],a.df[a.y])
        ax[i].plot(a.df[a.x],a.y_pre)
        # ax[i].plot(a.df[a.x], a.single1, alpha = 0.3)
        # ax[i].plot(a.df[a.x], a.single2, alpha = 0.3)

        returndata['f_FMR'].append(f)
        if peak_n == 1 :
            for name in ['K_s','H_res','delta_H','K_as','C']:
                returndata[name].append(a.returndata[name])

        elif peak_n == 2 :
            for name in ['K_s','H_res','delta_H','K_as','K_s2','H_res2','delta_H2','K_as2','C']:
                returndata[name].append(a.returndata[name])
        
    # print(returndata)
    returndata_df = pd.DataFrame(returndata)
    print('--------------------------------------------------------------')
    print('fit parameter:')
    print(returndata_df)


    #洛伦兹函数拟合完毕，开始拟合kittel和damping
    ax.append(fig.add_subplot(ax_line,ax_row,i+2))
    ax.append(fig.add_subplot(ax_line,ax_row,i+3))

    if peak_n == 1:
        for name in ['H_res','delta_H']:
            returndata_df[name] =  [Fit.change_unit(x,type='field') for x in returndata_df[name]] 
        returndata_df['f_FMR'] =  [Fit.change_unit(x,type='frequency') for x in returndata_df['f_FMR']] 
        returndata_df.to_csv('returnda.csv')

        print('--------------------------------------------------------------')
        print('fit parameter (SI unit):')
        print(returndata_df)

        b = Fit()
        f_pre = b.fit_Kittel_functions(returndata_df)
        H_pre = b.fit_damping_function(returndata_df)
        ax[i+1].plot(returndata_df['H_res'],f_pre,alpha=0.5)
        ax[i+1].scatter(returndata_df['H_res'],returndata_df['f_FMR'],alpha=0.5)
        ax[i+2].plot(returndata_df['f_FMR'],H_pre,alpha=0.5)
        ax[i+2].scatter(returndata_df['f_FMR'],np.abs(returndata_df['delta_H']),alpha=0.5)
        key_parameter = {}
        key_parameter['gamma (GHz T-1)'] = [b.gamma/(2*np.pi)*1E-9]
        key_parameter['H_k (Oe)'] = [b.H_k*4*np.pi*1E-3]
        key_parameter['miu_0*M_s (T)'] = [4*np.pi*1E-7*b.M_s]
        key_parameter['delta_H_0 (Oe)'] = [b.delta_H0*4*np.pi*1E-3]
        key_parameter['alpha'] = [b.damping]
        key_parameter = pd.DataFrame(key_parameter)
        key_parameter.to_csv('keyparameter.csv')

    elif peak_n == 2:
        for name in ['H_res','delta_H','H_res2','delta_H2']:
            returndata_df[name] =  [Fit.change_unit(x,type='field') for x in returndata_df[name]] 
        returndata_df['f_FMR'] =  [Fit.change_unit(x,type='frequency') for x in returndata_df['f_FMR']] 
        returndata_df.to_csv('returnda.csv')

        print('--------------------------------------------------------------')
        print('fit parameter (SI unit):')
        print(returndata_df)

        print('for the first peak')
        b = Fit()
        f_pre = b.fit_Kittel_functions(returndata_df)
        H_pre = b.fit_damping_function(returndata_df)
        ax[i+1].plot(returndata_df['H_res'],f_pre,alpha=0.5)
        ax[i+1].scatter(returndata_df['H_res'],returndata_df['f_FMR'],alpha=0.5)
        ax[i+2].plot(returndata_df['f_FMR'],H_pre,alpha=0.5)
        ax[i+2].scatter(returndata_df['f_FMR'],np.abs(returndata_df['delta_H']),alpha=0.5)
        key_parameter = {}
        key_parameter['gamma (GHz T-1)'] = [b.gamma/(2*np.pi)*1E-9]
        key_parameter['H_k (Oe)'] = [b.H_k*4*np.pi*1E-3]
        key_parameter['miu_0*M_s (T)'] = [4*np.pi*1E-7*b.M_s]
        key_parameter['delta_H_0 (Oe)'] = [b.delta_H0*4*np.pi*1E-3]
        key_parameter['alpha'] = [b.damping]

        print('--------------------------------------------------------------')
        print('for the second peak')
        b = Fit(H_res_n='H_res2',delta_H_n='delta_H2')
        f_pre = b.fit_Kittel_functions(returndata_df)
        H_pre = b.fit_damping_function(returndata_df)
        ax[i+1].plot(returndata_df['H_res2'],f_pre,alpha=0.5)
        ax[i+1].scatter(returndata_df['H_res2'],returndata_df['f_FMR'],alpha=0.5)
        ax[i+2].plot(returndata_df['f_FMR'],H_pre,alpha=0.5)
        ax[i+2].scatter(returndata_df['f_FMR'],np.abs(returndata_df['delta_H2']),alpha=0.5)
        key_parameter['gamma2 (GHz T-1)'] = [b.gamma/(2*np.pi)*1E-9]
        key_parameter['H_k2 (Oe)'] = [b.H_k*4*np.pi*1E-3]
        key_parameter['miu_0*M_s2 (T)'] = [4*np.pi*1E-7*b.M_s]
        key_parameter['delta_H_02 (Oe)'] = [b.delta_H0*4*np.pi*1E-3]
        key_parameter['alpha2'] = [b.damping]
        key_parameter = pd.DataFrame(key_parameter)
        key_parameter.to_csv('keyparameter.csv')

    plt.savefig('result.png',dpi=600)
    plt.show()
