from PyQt5 import QtWidgets
from ui.mainwindow import Ui_MainWindow  # 导入生成的界面类
from core.work import CroppingThread, ModelLoaderThread
import os
import webbrowser

class Setting:
    def __init__(self, ratio=1, direction=1, mode=1, input_path=None, output_path=None, number=None, start_time=None, end_time=None):
        """
         参数:
         - ratio: 比例参数，默认为 1。
         - direction: 方向参数，默认为 1。
         - mode: 模式参数，默认为 1。
         - input_path: 输入路径，默认为 None。
         - output_path: 输出路径，默认为 None。
         - number: 数量。
         - start_time: 开始时间，默认为 None。
         - end_time: 结束时间，默认为 None。
         """
        self.ratio = ratio
        self.direction = direction
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path
        self.number = number
        self.start_time = start_time
        self.end_time = end_time

userSetting = Setting()

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 初始化界面

        # 创建线程
        self.model_thread = ModelLoaderThread()
        self.model_thread.log_signal.connect(self.append_log)  # 连接日志信号
        self.model_thread.finished.connect(self.on_models_loaded)  # 连接完成信号
        self.model_thread.start()

        self.ok_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        self.cancel_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel)
        # 初始状态：取消按钮为灰色不可点击
        self.cancel_button.setEnabled(False)
        # 用于控制裁剪任务的标志
        self.is_running = False
        # 添加一个变量用于存储QThread对象
        self.cropping_thread = None

        #输入按钮点击
        self.pushButton_inchoose.clicked.connect(self.select_file_in)

        #尺寸选择
        self.ratioGroup = QtWidgets.QButtonGroup(self)
        self.ratioGroup.addButton(self.fiveinches, 1)
        self.ratioGroup.addButton(self.sixinches, 2)
        self.ratioGroup.addButton(self.sixinchesLong, 3)

        self.ratioGroup.buttonClicked.connect(self.on_radio_button_clicked)

        #方向选择
        self.directionGroup = QtWidgets.QButtonGroup(self)
        self.directionGroup.addButton(self.originalD, 1)
        self.directionGroup.addButton(self.verticalD, 2)
        self.directionGroup.addButton(self.horizonalD, 3)

        self.directionGroup.buttonClicked.connect(self.on_direction_button_clicked)

        #模式选择
        self.modeGroup = QtWidgets.QButtonGroup(self)
        self.modeGroup.addButton(self.smartCropping, 1)
        self.modeGroup.addButton(self.leaveBlank, 2)

        self.modeGroup.buttonClicked.connect(self.on_mode_button_clicked)

        # 输出按钮点击
        self.pushButton_outchoose.clicked.connect(self.select_file_out)

        # 打开文件夹按钮点击
        self.openfile.clicked.connect(self.open_folder)

        # 连接按钮信号到槽函数
        self.ok_button.clicked.connect(self.start_image_cropping)
        self.cancel_button.clicked.connect(self.cancel_image_cropping)

        self.append_log("就绪")
        self.plabel.setText("就绪")

    def on_models_loaded(self):
        self.append_log("所有模型已成功加载，程序就绪！")

    def disable_controls(self):
        """ 禁用相关的控件 """
        self.pushButton_inchoose.setEnabled(False)
        self.pushButton_outchoose.setEnabled(False)
        self.fiveinches.setEnabled(False)
        self.sixinches.setEnabled(False)
        self.sixinchesLong.setEnabled(False)
        self.originalD.setEnabled(False)
        self.verticalD.setEnabled(False)
        self.horizonalD.setEnabled(False)
        self.smartCropping.setEnabled(False)
        self.leaveBlank.setEnabled(False)

    def enable_controls(self):
        """ 启用相关的控件 """
        self.pushButton_inchoose.setEnabled(True)
        self.pushButton_outchoose.setEnabled(True)
        self.fiveinches.setEnabled(True)
        self.sixinches.setEnabled(True)
        self.sixinchesLong.setEnabled(True)
        self.originalD.setEnabled(True)
        self.verticalD.setEnabled(True)
        self.horizonalD.setEnabled(True)
        self.smartCropping.setEnabled(True)
        self.leaveBlank.setEnabled(True)


    def select_file_in(self):
        """
        输入文件选择
        """
        # 弹出文件选择对话框
        file_dialog_in = QtWidgets.QFileDialog(self)
        #file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)  # 选择文件
        file_dialog_in.setFileMode(QtWidgets.QFileDialog.Directory)  # 选择文件夹

        if file_dialog_in.exec_():
            selected_path = file_dialog_in.selectedFiles()[0]
            self.lineInPath.setText(selected_path)
            userSetting.input_path = selected_path

    def on_radio_button_clicked(self, button):
        """
        比例选择
        """
        if button == self.fiveinches:
            userSetting.ratio = 1
        elif button == self.sixinches:
            userSetting.ratio = 2
        elif button == self.sixinchesLong:
            userSetting.ratio = 3
        print(f"当前比例选择的值: {userSetting.ratio}")

    def on_direction_button_clicked(self, button):
        """
        方向选择
        """
        if button == self.originalD:
            userSetting.direction = 1
        elif button == self.verticalD:
            userSetting.direction = 2
        elif button == self.horizonalD:
            userSetting.direction = 3
        print(f"当前方向选择的值: {userSetting.direction}")

    def on_mode_button_clicked(self, button):
        """
        模式选择
        """
        if button == self.smartCropping:
            userSetting.mode = 1
        elif button == self.leaveBlank:
            userSetting.mode = 2
        print(f"当前模式选择的值: {userSetting.mode}")

    def select_file_out(self):
        """
        输出文件选择
        """
        # 弹出文件选择对话框
        file_dialog_out = QtWidgets.QFileDialog(self)
        file_dialog_out.setFileMode(QtWidgets.QFileDialog.Directory)  # 选择文件夹

        if file_dialog_out.exec_():
            selected_path = file_dialog_out.selectedFiles()[0]
            self.lineOutPath.setText(selected_path)
            userSetting.output_path = selected_path

    def start_image_cropping(self):
        """
        确认按钮点击事件：启动图像裁剪工作
        """
        if not userSetting.input_path:
            self.plabel.setText("请选择输入文件夹!")
            return
        elif not userSetting.output_path:
            self.plabel.setText("请选择输出文件夹!")
            return
        elif not os.path.exists(userSetting.input_path):
            self.plabel.setText("输入文件夹不存在！")
            return

        self.ok_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.is_running = True
        self.disable_controls()  # 禁用控件
        # 创建并启动QThread线程
        self.cropping_thread = CroppingThread(userSetting)
        self.cropping_thread.log_signal.connect(self.append_log)  # 连接信号到日志显示
        self.cropping_thread.plabel_signal.connect(self.update_label)  # 连接信号到 UI
        self.cropping_thread.finished_signal.connect(self.on_task_finished)  # 任务完成信号
        self.cropping_thread.progress_signal.connect(self.progressBar.setValue)  # 连接进度信号

        self.cropping_thread.start()

    def cancel_image_cropping(self):
        """
        取消按钮点击事件：取消正在运行的工作
        """
        self.plabel.setText("取消中……")
        if self.cropping_thread:
            self.cropping_thread.stop()
            self.cropping_thread.wait()  # 等待线程完全退出
            self.cropping_thread = None

        self.cancel_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.is_running = False
        self.progressBar.setValue(0)  # 重置进度条
        self.enable_controls()  # 启用控件


        self.plabel.setText("已取消")

    def on_task_finished(self):
        """ 任务完成后恢复控件 """
        self.is_running = False
        self.cancel_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.append_log("=======================================\n\n")
        self.enable_controls()  # 启用控件
        self.plabel.setText("任务完成")

    def append_log(self, message):
        """
        追加日志信息到文本框
        """
        self.logText.append(message)
        self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())

    def update_label(self, message):
        """ 用于更新 pLabel 的槽函数 """
        self.plabel.setText(message)  # 只能在主线程里修改 UI

    def open_folder(self):
        if userSetting.output_path:
            folder_path = userSetting.output_path.strip()
        else:
            folder_path = os.path.join(os.path.expanduser("~"), "Desktop")  # 默认打开桌面

        if os.path.exists(folder_path):  # 确保路径存在
            webbrowser.open(os.path.abspath(folder_path))  # 直接打开文件夹
        else:
            print("错误：指定的文件夹不存在！")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)  # 创建应用程序对象
    window = MyApp()  # 创建主窗口对象
    window.show()  # 显示窗口
    sys.exit(app.exec_())  # 运行应用程序主循环