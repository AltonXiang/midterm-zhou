# Pipeline for Task 1

### Step1：预训练，check大致效果

```python
python pre_main.py
```

### Step2：比较数据增强前后效果

```
final_results.ipynb(Best Model之前)
```

### Step3：调参之后，加深网络，继续训练

```python
python main.py --aug_type baseline --save_path ./models/baseline.pth
python main.py --aug_type cutmix --save_path ./models/cutmix.pth
python main.py --aug_type cutout --save_path ./models/cutout.pth
python main.py --aug_type mixup --save_path ./models/mixup.pth
```

模型请于此处下载：链接：https://pan.baidu.com/s/1a6bMq9tDNH7FMOyPO6P8oA  提取码：oosl 

下载之后保存在./models

### Step4：比较三种数据增强以及baseline的性能

```
final_results.ipynb(Best Model部分)
```

