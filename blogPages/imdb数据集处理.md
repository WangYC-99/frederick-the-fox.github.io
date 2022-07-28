# 数据集处理常用tips

> by WangYC
>
> @NWPU changan Apr.4th 2022

### 去掉string中的空格以及换行等分隔符

https://zhuanlan.zhihu.com/p/44342284

### 读txt为numpy数组

https://www.yisu.com/zixun/171317.html

### 给df中某一个位置的内容赋值

https://www.cnblogs.com/wodexk/p/10316793.html

### df遍历行值

https://blog.csdn.net/d1240673769/article/details/112407978

### python逐行写入txt文件

https://blog.csdn.net/qq_37730871/article/details/116896097

### 判断一个浮点数是否为NaN

```python
math.isnan(row['movie_num']) is False
```

### python zip 函数

https://www.cnblogs.com/anita-harbour/p/9328597.html#:~:text=python%E5%9F%BA%E7%A1%80%EF%BC%9Azip%E5%92%8Cdict%E8%AF%A6%E8%A7%A3%20%E4%B8%80.zip%E5%87%BD%E6%95%B0%EF%BC%9A%E6%8E%A5%E5%8F%97%E4%BB%BB%E6%84%8F%E5%A4%9A%E4%B8%AA%EF%BC%88%E5%8C%85%E6%8B%AC0%E4%B8%AA%E5%92%8C1%E4%B8%AA%EF%BC%89%E5%BA%8F%E5%88%97%E4%BD%9C%E4%B8%BA%E5%8F%82%E6%95%B0%EF%BC%8C%E8%BF%94%E5%9B%9E%E4%B8%80%E4%B8%AAtuple%E5%88%97%E8%A1%A8%E3%80%82%201.%E7%A4%BA%E4%BE%8B1%EF%BC%9A%20x%20%3D%20%5B1%2C%202%2C%203%5D,xyz%20%3D%20zip%20%28x%2C%20y%2C%20z%29%20print%20xyz

### ndarray 与 scipy.sparse.csr.csr_matrix 的互转

https://blog.csdn.net/Scythe666/article/details/84623786

### 读pkl文件

https://blog.csdn.net/King_key/article/details/99778042

### 读npy/npz文件

https://blog.csdn.net/qq_33254870/article/details/90675439

npy是单个矩阵的保存，npz是一个类似于压缩文件一样的npy集合

load的时候npz.files可以看内容的list；npz[label]查看具体的向量

### numpy数组的reshape

https://blog.csdn.net/qq_42804678/article/details/99062431

 