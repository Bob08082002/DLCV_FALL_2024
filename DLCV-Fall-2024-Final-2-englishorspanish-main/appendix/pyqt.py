import random
import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QInputDialog, QMessageBox
########################################
# PyQt Integration Code
########################################
class BoundingBoxSelector(QMainWindow):
    def __init__(self, num, W, H):
        super().__init__()
        self.setWindowTitle("Bounding Box Selector")
        self.setGeometry(100, 100, W, H)  # Set canvas size to WxH

        self.num = num  # Maximum number of bounding boxes
        self.current_count = 0  # Current number of bounding boxes

        self.start_point = None
        self.end_point = None
        self.bounding_box = None

        # Create a black image as background
        self.image = QImage(W, H, QImage.Format_RGB888)
        self.image.fill(QColor('black'))

        self.bbx_list = []

        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 492, 20)
        self.label.setStyleSheet("color: red;")
        self.label.setWordWrap(True)
        self.label.raise_()

    def map_to_image(self, point):
        # Since image and canvas size are the same, no scaling needed
        return point.x(), point.y()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.current_count >= self.num:
                self.show_max_reached_dialog()
                return
            self.start_point = event.pos()
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        if self.start_point and event.buttons() == Qt.LeftButton:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            if self.current_count >= self.num:
                self.show_max_reached_dialog()
                return

            self.end_point = event.pos()
            mapped_start = self.map_to_image(self.start_point)
            mapped_end = self.map_to_image(self.end_point)

            ymin = min(mapped_start[1], mapped_end[1])
            xmin = min(mapped_start[0], mapped_end[0])
            ymax = max(mapped_start[1], mapped_end[1])
            xmax = max(mapped_start[0], mapped_end[0])

            self.bounding_box = (ymin, xmin, ymax, xmax)
            self.current_count += 1
            self.bbx_list.append(self.bounding_box)

            print(f"Bounding Box: ({ymin}, {xmin}, {ymax}, {xmax})")
            self.label.setText(f"Bounding Box: (ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax})")

            self.start_point = None
            self.end_point = None
            self.update()

            # If we have reached the required number of boxes, show dialog
            if self.current_count == self.num:
                self.show_max_reached_dialog()

    def show_max_reached_dialog(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Maximum Bounding Boxes Reached")
        dialog.setText("You have reached the maximum number of bounding boxes.")
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setDefaultButton(QMessageBox.Ok)

        choice = dialog.exec_()

        if choice == QMessageBox.Ok:
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.image.isNull():
            painter.drawImage(0, 0, self.image)

        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        for box in self.bbx_list:
            ymin, xmin, ymax, xmax = box
            painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)

        if self.start_point and self.end_point:
            temp_start = self.map_to_image(self.start_point)
            temp_end = self.map_to_image(self.end_point)
            temp_ymin = min(temp_start[1], temp_end[1])
            temp_xmin = min(temp_start[0], temp_end[0])
            temp_ymax = max(temp_start[1], temp_end[1])
            temp_xmax = max(temp_start[0], temp_end[0])
            painter.drawRect(temp_xmin, temp_ymin, temp_xmax - temp_xmin, temp_ymax - temp_ymin)


def get_user_defined_bboxes(num_bboxes, W, H):
    # Run a PyQt Application to get bounding boxes
    app = QApplication(sys.argv)
    window = BoundingBoxSelector(num_bboxes, W, H)
    window.show()
    app.exec_()  # block until user closes

    # window.bbx_list contains [(ymin,xmin,ymax,xmax), ...]
    return window.bbx_list


target_bboxes_user = get_user_defined_bboxes(2, 512, 512)
