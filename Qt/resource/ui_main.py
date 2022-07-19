# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(821, 602)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(20, 10, 781, 581))
        self.correct_answe_uploading_tab = QWidget()
        self.correct_answe_uploading_tab.setObjectName(u"correct_answe_uploading_tab")
        self.verticalLayoutWidget = QWidget(self.correct_answe_uploading_tab)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(9, 9, 761, 541))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.file_chooser_layout = QVBoxLayout()
        self.file_chooser_layout.setObjectName(u"file_chooser_layout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.CAU_file_select_lineEdit = QLineEdit(self.verticalLayoutWidget)
        self.CAU_file_select_lineEdit.setObjectName(u"CAU_file_select_lineEdit")

        self.horizontalLayout.addWidget(self.CAU_file_select_lineEdit)

        self.CAU_file_select_pushButton = QPushButton(self.verticalLayoutWidget)
        self.CAU_file_select_pushButton.setObjectName(u"CAU_file_select_pushButton")

        self.horizontalLayout.addWidget(self.CAU_file_select_pushButton)


        self.file_chooser_layout.addLayout(self.horizontalLayout)


        self.verticalLayout.addLayout(self.file_chooser_layout)

        self.verticalSpacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.uploading_layout = QHBoxLayout()
        self.uploading_layout.setObjectName(u"uploading_layout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.uploading_layout.addItem(self.horizontalSpacer)

        self.CAU_uploading_pushButton = QPushButton(self.verticalLayoutWidget)
        self.CAU_uploading_pushButton.setObjectName(u"CAU_uploading_pushButton")

        self.uploading_layout.addWidget(self.CAU_uploading_pushButton)


        self.verticalLayout.addLayout(self.uploading_layout)

        self.verticalSpacer_2 = QSpacerItem(20, 80, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.file_chooser_layout_2 = QVBoxLayout()
        self.file_chooser_layout_2.setObjectName(u"file_chooser_layout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.CAU_file_select_lineEdit_2 = QLineEdit(self.verticalLayoutWidget)
        self.CAU_file_select_lineEdit_2.setObjectName(u"CAU_file_select_lineEdit_2")

        self.horizontalLayout_2.addWidget(self.CAU_file_select_lineEdit_2)

        self.CAU_file_select_pushButton_2 = QPushButton(self.verticalLayoutWidget)
        self.CAU_file_select_pushButton_2.setObjectName(u"CAU_file_select_pushButton_2")

        self.horizontalLayout_2.addWidget(self.CAU_file_select_pushButton_2)


        self.file_chooser_layout_2.addLayout(self.horizontalLayout_2)


        self.verticalLayout.addLayout(self.file_chooser_layout_2)

        self.verticalSpacer_3 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.uploading_layout_2 = QHBoxLayout()
        self.uploading_layout_2.setObjectName(u"uploading_layout_2")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.uploading_layout_2.addItem(self.horizontalSpacer_2)

        self.CAU_uploading_pushButton_2 = QPushButton(self.verticalLayoutWidget)
        self.CAU_uploading_pushButton_2.setObjectName(u"CAU_uploading_pushButton_2")

        self.uploading_layout_2.addWidget(self.CAU_uploading_pushButton_2)


        self.verticalLayout.addLayout(self.uploading_layout_2)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_4)

        self.tabWidget.addTab(self.correct_answe_uploading_tab, "")
        self.answer_sheet_grading_tab = QWidget()
        self.answer_sheet_grading_tab.setObjectName(u"answer_sheet_grading_tab")
        self.verticalLayoutWidget_2 = QWidget(self.answer_sheet_grading_tab)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(10, 10, 761, 541))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.file_chooser_layout_3 = QVBoxLayout()
        self.file_chooser_layout_3.setObjectName(u"file_chooser_layout_3")
        self.file_chooser_layout_3.setSizeConstraint(QLayout.SetMinimumSize)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_3 = QLabel(self.verticalLayoutWidget_2)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.ASG_file_select_lineEdit = QLineEdit(self.verticalLayoutWidget_2)
        self.ASG_file_select_lineEdit.setObjectName(u"ASG_file_select_lineEdit")

        self.horizontalLayout_3.addWidget(self.ASG_file_select_lineEdit)

        self.ASG_file_select_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.ASG_file_select_pushButton.setObjectName(u"ASG_file_select_pushButton")

        self.horizontalLayout_3.addWidget(self.ASG_file_select_pushButton)


        self.file_chooser_layout_3.addLayout(self.horizontalLayout_3)


        self.verticalLayout_3.addLayout(self.file_chooser_layout_3)

        self.grading_layout = QHBoxLayout()
        self.grading_layout.setObjectName(u"grading_layout")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.grading_layout.addItem(self.horizontalSpacer_3)

        self.ASG_grading_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.ASG_grading_pushButton.setObjectName(u"ASG_grading_pushButton")

        self.grading_layout.addWidget(self.ASG_grading_pushButton)


        self.verticalLayout_3.addLayout(self.grading_layout)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.ASG_grading_status_tableWidget = QTableWidget(self.verticalLayoutWidget_2)
        self.ASG_grading_status_tableWidget.setObjectName(u"ASG_grading_status_tableWidget")

        self.verticalLayout_4.addWidget(self.ASG_grading_status_tableWidget)


        self.verticalLayout_3.addLayout(self.verticalLayout_4)

        self.tabWidget.addTab(self.answer_sheet_grading_tab, "")
        self.check_grade_result_tab = QWidget()
        self.check_grade_result_tab.setObjectName(u"check_grade_result_tab")
        self.verticalLayoutWidget_3 = QWidget(self.check_grade_result_tab)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(10, 10, 761, 541))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.CGR_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.CGR_pushButton.setObjectName(u"CGR_pushButton")

        self.verticalLayout_2.addWidget(self.CGR_pushButton)

        self.CGR_tableWidget = QTableWidget(self.verticalLayoutWidget_3)
        self.CGR_tableWidget.setObjectName(u"CGR_tableWidget")

        self.verticalLayout_2.addWidget(self.CGR_tableWidget)

        self.tabWidget.addTab(self.check_grade_result_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\uc815\ub2f5 \ud30c\uc77c: ", None))
        self.CAU_file_select_pushButton.setText(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c \uc120\ud0dd", None))
        self.CAU_uploading_pushButton.setText(QCoreApplication.translate("MainWindow", u"\uc5c5\ub85c\ub4dc", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\uc6d0\ubcf8 \uc774\ubbf8\uc9c0: ", None))
        self.CAU_file_select_pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c \uc120\ud0dd", None))
        self.CAU_uploading_pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\uc5c5\ub85c\ub4dc", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.correct_answe_uploading_tab), QCoreApplication.translate("MainWindow", u"\uc815\ub2f5 \uc5c5\ub85c\ub4dc", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\ub2f5\uc548 \uc774\ubbf8\uc9c0: ", None))
        self.ASG_file_select_pushButton.setText(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c \uc120\ud0dd", None))
        self.ASG_grading_pushButton.setText(QCoreApplication.translate("MainWindow", u"\ucc44\uc810", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.answer_sheet_grading_tab), QCoreApplication.translate("MainWindow", u"\ucc44\uc810", None))
        self.CGR_pushButton.setText(QCoreApplication.translate("MainWindow", u"\ucc44\uc810 \uacb0\uacfc \uac00\uc838\uc624\uae30", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.check_grade_result_tab), QCoreApplication.translate("MainWindow", u"\ucc44\uc810 \uacb0\uacfc \ud655\uc778", None))
    # retranslateUi

