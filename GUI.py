# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(512, 397)
        self.pBt_import = QtWidgets.QPushButton(Form)
        self.pBt_import.setGeometry(QtCore.QRect(30, 20, 93, 28))
        self.pBt_import.setObjectName("pBt_import")
        self.lineEdit_path = QtWidgets.QLineEdit(Form)
        self.lineEdit_path.setGeometry(QtCore.QRect(210, 20, 271, 31))
        self.lineEdit_path.setObjectName("lineEdit_path")
        self.lbl_path = QtWidgets.QLabel(Form)
        self.lbl_path.setGeometry(QtCore.QRect(140, 30, 72, 15))
        self.lbl_path.setObjectName("lbl_path")
        self.pBt_classfy = QtWidgets.QPushButton(Form)
        self.pBt_classfy.setGeometry(QtCore.QRect(360, 220, 93, 28))
        self.pBt_classfy.setObjectName("pBt_classfy")
        self.lbl_result = QtWidgets.QLabel(Form)
        self.lbl_result.setGeometry(QtCore.QRect(340, 280, 72, 15))
        self.lbl_result.setObjectName("lbl_result")
        self.lineEdit_result = QtWidgets.QLineEdit(Form)
        self.lineEdit_result.setGeometry(QtCore.QRect(410, 270, 81, 31))
        self.lineEdit_result.setObjectName("lineEdit_result")
        self.lbl_acc = QtWidgets.QLabel(Form)
        self.lbl_acc.setGeometry(QtCore.QRect(340, 330, 72, 15))
        self.lbl_acc.setObjectName("lbl_acc")
        self.lineEdit_acc = QtWidgets.QLineEdit(Form)
        self.lineEdit_acc.setGeometry(QtCore.QRect(410, 320, 81, 31))
        self.lineEdit_acc.setObjectName("lineEdit_acc")
        self.lbl_fig = QtWidgets.QLabel(Form)
        self.lbl_fig.setGeometry(QtCore.QRect(40, 80, 251, 251))
        self.lbl_fig.setText("")
        self.lbl_fig.setObjectName("lbl_fig")
        self.lbl_fig.setStyleSheet("QLabel{\n"
                                 "    border-color: rgb(255, 170,0);\n"
                                 "     border-width: 1px;\n"
                                 "     border-style: solid;\n"
                                 "}")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.pBt_import.clicked.connect(Form.openimage)
        self.pBt_classfy.clicked.connect(Form.classfy)
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pBt_import.setText(_translate("Form", "导入图片"))
        self.lbl_path.setText(_translate("Form", "图片路径"))
        self.pBt_classfy.setText(_translate("Form", "识别分类"))
        self.lbl_result.setText(_translate("Form", "识别结果"))
        self.lbl_acc.setText(_translate("Form", "可信度"))
