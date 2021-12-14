# Fruit-Classifier-by-CNN（机器视觉及应用作业）
基于CNN的水果分类器 (Pytorch)：下载后打开main_window.py运行可以看到test.gif的形式，分类包括11种水果：(a)苹果;(b)李子;(c)樱桃;(d)圣女果;(e)山楂;(f)青枣;(g)猕猴桃;(h)柠檬;(i)龙眼;(j)青橘;(k)红橘

（已改成相对路径不需设置其他地方，但需要安装如下**环境**）

## 环境
#### 编程语言
+ Python>=3.9.2
#### Pytorch
+ torch==1.8.0+cu111 
+ torchvision==0.9.0+cu111 
如果你要使用同样的torch库（我选择这个库是因为我的GPU驱动版本是11.1的），你可以在命令行窗口运行下面的pip命令（一键配置大概3个多G）：
>pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html

你也可以使用更高版本的torch，但不能保证程序能运行成功
#### GUI界面制作
+ pyqt==5.15.2
同样可以使用pip进行安装
>pip install PyQt5
>pip install PyQt5-tools

## 文件说明
**data/images**：水果图片

**data/labelnum.txt**：各种水果的标号、名称、数量

**data/test.txt**：从images中挑选的测试集

**data/train.txt**：从images中挑选的训练集

**CNN_Data_inout.py**:图片文件（jpg）、txt文件等的读取、写入模块

**CNN_fruit_train_test.py**：CNN测试和训练模块，输出结果保存为model_val.pth

**CNN_fruit_final.py**：上一文件输出的最终模型模块model_val.pth（可直接判别）

**CNN_model.py**：建立的CNN模型，使用了AlexNet,ResNet，Inception框架进行组合

**GUI.py**：GUI界面布局

**main_window.py**：GUI主程序，包括回调槽函数
