from GUI import *
import sys
from GUI import Ui_Form   # 导入生成first.py里生成的类
from PyQt5.QtWidgets import QFileDialog
import CNN_fruit_final

class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #定义槽函数
    def openimage(self):
   # 打开文件路径
   #设置文件扩展名过滤,注意用双分号间隔
        imgName=''
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    "",
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")


        #利用qlabel显示图片
        self.lineEdit_path.setText(imgName)
        self.imgpath=[imgName]

        png = QtGui.QPixmap(imgName).scaled(self.lbl_fig.width(), self.lbl_fig.height())
        self.lbl_fig.setPixmap(png)

    def classfy(self):
        pred_infor, score_lists = CNN_fruit_final.classfy(self.imgpath)
        pred_infor_Chinese={
                               'Apple':'苹果',
                               'Brin':'李子',
                               'Cherry':'樱桃',
                               'Cherry_Tomatoes':'圣女果',
                               'Hawthorn':'山楂',
                               'Jujube':'青枣',
                               'Kiwi':'猕猴挑',
                               'Lemon':'柠檬',
                               'Longan':'龙眼',
                               'Orange_green':'青橘',
                               'Orange_red':'红橘',
                                }
        self.lineEdit_result.setText(pred_infor_Chinese[str(pred_infor[0])])
        self.lineEdit_acc.setText(str(format(score_lists[0],'.6f')))

if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
