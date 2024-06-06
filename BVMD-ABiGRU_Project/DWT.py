import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family']='Times New Roman ,SimSun '

data = pd.read_csv('data/demo.csv', encoding='gbk')
signal = data[data.columns[2]].values
# signal = signal[:200]
t = np.linspace(1, len(signal), len(signal))

# 常见的几种小波基函数包括：

# 1. Daubechies小波基（db）：Daubechies小波基是最常用的小波基函数之一。它具有紧凑支持和良好的频率局部化特性。常见的Daubechies小波基包括db2、db4、db6等。

# 2. Symlets小波基（sym）：Symlets小波基是对称的Daubechies小波基。它们在频率局部化和相位对称性方面与Daubechies小波基类似。常见的Symlets小波基包括sym2、sym4、sym8等。

# 3. Coiflets小波基（coif）：Coiflets小波基是具有紧凑支持和较好频率局部化特性的小波基。它们在一些应用中比Daubechies小波基具有更好的性能。常见的Coiflets小波基包括coif1、coif2、coif3等。

# 4. Biorthogonal小波基（bior）：Biorthogonal小波基是一组成对的小波基函数。它们具有可变的支持长度和频率响应。常见的Biorthogonal小波基包括bior2.2、bior3.3、bior6.8等。

wavelet = 'db2'  # 定义小波基名称为'db4'
# wavelet_name = 'sym4'  # 定义小波基名称为'sym4'
# wavelet_name = 'bior3.3'  # 定义小波基名称为'bior3.3'

# 小波变换
coeffs = pywt.wavedec(signal, wavelet, level=3)  # 使用指定小波基进行4级小波分解



cA3 = coeffs[0]
cD3 = coeffs[1]
cD2 = coeffs[2]
cD1 = coeffs[3]
cA3 = pywt.upcoef('a', cA3, wavelet, level=3)[:len(signal)]
cD3 = pywt.upcoef('d', cD3, wavelet, level=3)[:len(signal)]
cD2 = pywt.upcoef('d', cD2, wavelet, level=3)[:len(signal)]
cD1 = pywt.upcoef('d', cD1, wavelet, level=3)[:len(signal)]


plt.figure(figsize=(6, 6))
# plt.subplot(5, 1, 1)
# plt.plot(t, signal)
# plt.title('Original Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')


plt.subplot(4, 1, 1)
plt.plot(t[:len(cA3)], cA3)
plt.tick_params(axis='both', direction='in')

plt.subplot(4, 1, 2)
plt.plot(t[:len(cD3)], cD3)
plt.tick_params(axis='both', direction='in')

plt.subplot(4, 1, 3)
plt.plot(t[:len(cD2)], cD2)
plt.tick_params(axis='both', direction='in')

plt.subplot(4, 1, 4)
plt.plot(t[:len(cD1)], cD1)
plt.tick_params(axis='both', direction='in')


merged_df = pd.concat([pd.DataFrame(cA3),pd.DataFrame(cD3),pd.DataFrame(cD2),pd.DataFrame(cD1)], axis=1)
merged_df.to_csv('./data/imfst.csv', mode='w', index=False, encoding='gbk')


plt.tight_layout()
plt.savefig('xiaobo.svg', dpi=600, format="svg")
# plt.show()