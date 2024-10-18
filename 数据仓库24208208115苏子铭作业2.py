import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('震动测试数据曲线.xlsx')

# 选择时间和震动源进行特征提取
t = data['采样时间(s)']
signal = data['输出电压（V）']

# DFT
dft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))

# 小波变换
wavelet = 'db4'  # 选择小波基
coeffs = pywt.wavedec(signal, wavelet)
detail_coeffs = coeffs[1:]  # 细节系数
approx_coeffs = coeffs[0]  # 逼近系数

# 多项式拟合
degree = 5
poly_coeffs = np.polyfit(t, signal, degree)
poly_fit = np.polyval(poly_coeffs, t)

# 画图展示
plt.figure(figsize=(12, 10))

# 原始信号
plt.subplot(4, 1, 1)
plt.plot(t, signal, label='原始信号', color='blue')
plt.title('原始信号')
plt.xlabel('时间 (秒)')
plt.ylabel('输出电压（V）')
plt.grid()
plt.legend()
plt.text(t.iloc[len(t)//2], signal.max()*0.8, '原始信号', fontsize=10, color='blue')

# DFT结果
plt.subplot(4, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft_result)[:len(dft_result)//2], label='DFT结果', color='orange')
plt.title('DFT结果')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.grid()
plt.legend()
plt.text(frequencies[len(frequencies)//4], np.abs(dft_result).max()*0.8, 'DFT结果', fontsize=10, color='orange')

# 小波逼近系数
plt.subplot(4, 1, 3)
plt.plot(t, np.repeat(approx_coeffs, len(signal)//len(approx_coeffs)), label='小波逼近系数', color='green')
plt.title('小波变换逼近系数')
plt.xlabel('时间 (秒)')
plt.ylabel('逼近系数值')
plt.grid()
plt.legend()
plt.text(t.iloc[len(t)//2], approx_coeffs.max()*0.8, '小波逼近系数', fontsize=10, color='green')

# 多项式拟合图
plt.subplot(4, 1, 4)
plt.plot(t, signal, label='原始信号', color='blue')
plt.plot(t, poly_fit, label='多项式拟合', color='red', linestyle='--')
plt.title('多项式拟合')
plt.xlabel('时间 (秒)')
plt.ylabel('输出电压（V）')
plt.grid()
plt.legend()
plt.text(t.iloc[len(t)//2], poly_fit.max()*0.8, '多项式拟合', fontsize=10, color='red')

plt.tight_layout()
plt.show()

# 特征比较
mse_poly = np.mean((signal - poly_fit) ** 2)
mse_dft = np.mean(np.abs(dft_result))
mse_wavelet = np.mean(np.abs(approx_coeffs))

print(f'多项式拟合均方误差: {mse_poly}')
print(f'DFT均值幅值: {mse_dft}')
print(f'小波逼近系数均值: {mse_wavelet}')

if mse_poly < mse_dft and mse_poly < mse_wavelet:
    print('多项式拟合效果最好。')
elif mse_dft < mse_poly and mse_dft < mse_wavelet:
    print('DFT效果最好。')
else:
    print('小波变换效果最好。')
