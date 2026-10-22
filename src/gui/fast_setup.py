"""Native desktop workspace for fast wall, door, and light setup."""

from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from src.core.fast_pipeline import (
    DetectionCancelled, FastSettings, detect_map, export_image_bytes, load_map,
    to_uvtt, write_uvtt,
)


class DetectionWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, source, path, settings, generation, parent=None):
        super().__init__(parent)
        self.image, self.source, self.path = image, source, path
        self.settings, self.generation = settings, generation

    def run(self):
        try:
            image, source = self.image, self.source
            if self.path:
                self.progress.emit("Loading map…")
                image, source = load_map(self.path)
            result = detect_map(image, self.settings, self.progress.emit, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.completed.emit((image, source, result, self.settings, self.generation))
        except DetectionCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class ExportWorker(QThread):
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, path, image, source, result, settings, parent=None):
        super().__init__(parent)
        self.path, self.image, self.source = path, image, source
        self.result, self.settings = deepcopy(result), settings

    def run(self):
        try:
            encoded = export_image_bytes(self.image, self.source)
            data = to_uvtt(self.result, self.settings.grid_size, encoded)
            write_uvtt(self.path, data)
            self.completed.emit(self.path)
        except Exception as error:
            self.failed.emit(str(error))


class MapView(QGraphicsView):
    """Keep the original bitmap cached; pan/zoom redraw only visible vector items."""

    color_picked = pyqtSignal(int, int)
    candidate_clicked = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#14171b"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.bitmap = None
        self.layers = {"walls": [], "doors": [], "lights": []}
        self.visibility = {name: True for name in self.layers}
        self.picking = False
        self.removing = False
        self.image_size = (0, 0)

    def set_image(self, image):
        # QPixmap owns the converted image data after fromImage returns.
        height, width = image.shape[:2]
        qt_image = QImage(image.data, width, height, image.strides[0], QImage.Format.Format_BGR888)
        self.scene().clear()
        self.layers = {name: [] for name in self.layers}
        self.bitmap = self.scene().addPixmap(QPixmap.fromImage(qt_image))
        self.bitmap.setZValue(-1)
        self.image_size = (width, height)
        self.setSceneRect(0, 0, width, height)
        self.fit_map()

    def fit_map(self):
        if self.bitmap:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def pen(color, width):
        pen = QPen(QColor(color), width)
        pen.setCosmetic(True)
        return pen

    def set_result(self, result):
        for items in self.layers.values():
            for item in items:
                self.scene().removeItem(item)
        self.layers = {name: [] for name in self.layers}
        path = QPainterPath()
        for start, end in result.walls:
            path.moveTo(*start)
            path.lineTo(*end)
        self.layers["walls"].append(self.scene().addPath(path, self.pen("#4ee19b", 1.6)))
        for index, (start, end) in enumerate(result.doors):
            item = self.scene().addLine(*start, *end, self.pen("#35d4ff", 4))
            item.setData(0, "doors")
            item.setData(1, index)
            item.setToolTip(f"Door {index + 1}")
            self.layers["doors"].append(item)
        radius = max(result.width, result.height) / 300
        for index, light in enumerate(result.lights):
            x, y = light["position"]["x"], light["position"]["y"]
            item = self.scene().addEllipse(x - radius, y - radius, 2 * radius, 2 * radius,
                                          self.pen("#ffd75e", 2.4))
            item.setData(0, "lights")
            item.setData(1, index)
            item.setToolTip(f"Light {index + 1}")
            self.layers["lights"].append(item)
        for name, visible in self.visibility.items():
            self.show_layer(name, visible)

    def show_layer(self, name, visible):
        self.visibility[name] = visible
        for item in self.layers[name]:
            item.setVisible(visible)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        zoom = self.transform().m11() * factor
        if 0.025 <= zoom <= 20:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.picking:
                if 0 <= point.x() < self.image_size[0] and 0 <= point.y() < self.image_size[1]:
                    self.color_picked.emit(int(point.x()), int(point.y()))
                event.accept()
                return
            if self.removing:
                for item in self.items(event.position().toPoint()):
                    if item.data(0) in ("doors", "lights"):
                        self.candidate_clicked.emit(item.data(0), item.data(1))
                        break
                event.accept()
                return
        super().mousePressEvent(event)


class FastSetupDialog(QDialog):
    """Single-flight workers, with newer requests replacing obsolete work."""

    def __init__(self, parent=None, image=None, source=None, path=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Wall • Fast Setup")
        self.resize(1380, 900)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.image, self.source, self.path = image, source, path
        self.result = None
        self.result_settings = None
        self.worker = None
        self.export_worker = None
        self.pending = False
        self.generation = 0
        self.closing = False
        self.pick_target = None
        self.wall_color = (131, 131, 131)
        self.door_color = (255, 160, 65)
        self.undo_states = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.run_detection)
        self._build_ui()
        if image is not None:
            self.view.set_image(image)
        if image is not None or path:
            QTimer.singleShot(0, self.run_detection)

    def _build_ui(self):
        root = QHBoxLayout(self)
        sidebar = QWidget()
        sidebar.setFixedWidth(325)
        controls = QVBoxLayout(sidebar)
        root.addWidget(sidebar)
        title = QLabel("FAST SETUP")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        controls.addWidget(title)
        caption = QLabel("Walls, doors, and lights in one pass.\nScroll to zoom. Drag to pan.")
        controls.addWidget(caption)
        open_button = QPushButton("Open map…")
        open_button.clicked.connect(self.open_map)
        controls.addWidget(open_button)
        common = QFormLayout()
        self.quality = QComboBox()
        for label, value in (("Fast · 2048 px", 2048), ("Detailed · 3072 px", 3072),
                             ("Fine · 4096 px", 4096), ("Original pixels", 0)):
            self.quality.addItem(label, value)
        self.quality.currentIndexChanged.connect(self.settings_changed)
        common.addRow("Detection", self.quality)
        self.grid = self._number(1, 4000, 70, decimals=1)
        common.addRow("Pixels / grid", self.grid)
        controls.addLayout(common)
        note = QLabel("Detection resolution never resizes your exported map. Grid size changes VTT units only.")
        note.setWordWrap(True)
        controls.addWidget(note)
        self.tabs = QTabWidget()
        controls.addWidget(self.tabs)
        walls = QFormLayout()
        self.wall_mode = QComboBox()
        for label, value in (("Auto stone / gray", "stone"), ("Sampled color", "color"), ("Edges", "edges")):
            self.wall_mode.addItem(label, value)
        self.wall_mode.currentIndexChanged.connect(self.settings_changed)
        walls.addRow("Method", self.wall_mode)
        self.wall_swatch = self._color_controls("wall", walls)
        self.wall_tolerance = self._number(0, 100, 24)
        walls.addRow("Color tolerance", self.wall_tolerance)
        self.min_area = self._number(0, 5000, 1000, decimals=1)
        walls.addRow("Minimum area", self.min_area)
        self.simplify = self._number(0, 10, 1.25, decimals=2)
        walls.addRow("Simplify", self.simplify)
        self._add_tab("Walls", walls)
        doors = QFormLayout()
        self.enable_doors = QCheckBox("Auto-detect doors")
        self.enable_doors.setChecked(True)
        self.enable_doors.toggled.connect(self.settings_changed)
        doors.addRow(self.enable_doors)
        self.door_swatch = self._color_controls("door", doors)
        self.hue = self._number(0, 90, 16)
        doors.addRow("Hue tolerance", self.hue)
        self.min_door = self._number(2, 400, 12, decimals=1)
        self.max_door = self._number(2, 800, 60, decimals=1)
        doors.addRow("Min length", self.min_door)
        doors.addRow("Max length", self.max_door)
        self.rectangularity = self._number(0.1, 1, 0.78, decimals=2)
        doors.addRow("Rectangularity", self.rectangularity)
        self.support = QCheckBox("Require nearby wall jambs")
        self.support.setChecked(True)
        self.support.toggled.connect(self.settings_changed)
        doors.addRow(self.support)
        self._add_tab("Doors", doors)
        lights = QFormLayout()
        self.enable_lights = QCheckBox("Auto-detect lights")
        self.enable_lights.setChecked(True)
        self.enable_lights.toggled.connect(self.settings_changed)
        lights.addRow(self.enable_lights)
        self.brightness = self._number(0, 255, 190)
        self.saturation = self._number(0, 255, 60)
        self.min_light = self._number(1, 5000, 3, decimals=1)
        self.max_light = self._number(1, 10000, 180, decimals=1)
        self.light_range = self._number(0.1, 100, 6, decimals=1)
        lights.addRow("Brightness", self.brightness)
        lights.addRow("Min saturation", self.saturation)
        lights.addRow("Min area", self.min_light)
        lights.addRow("Max area", self.max_light)
        lights.addRow("Range (squares)", self.light_range)
        self._add_tab("Lights", lights)
        units = QLabel("Lengths and areas use a 2048 px reference so settings remain consistent across quality levels.")
        units.setWordWrap(True)
        controls.addWidget(units)
        self.live = QCheckBox("Update automatically")
        self.live.setChecked(True)
        self.live.toggled.connect(self.settings_changed)
        controls.addWidget(self.live)
        row = QHBoxLayout()
        self.run_button = QPushButton("Detect all")
        self.run_button.clicked.connect(self.run_detection)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_detection)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        controls.addLayout(row)
        self.status = QLabel("Open a map to begin.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        controls.addWidget(self.status)
        controls.addStretch()
        self.export_button = QPushButton("Export UVTT…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_map)
        controls.addWidget(self.export_button)
        self.editor_button = QPushButton("Continue in wall editor")
        self.editor_button.setEnabled(False)
        self.editor_button.setVisible(self.parent() is not None)
        self.editor_button.clicked.connect(self.use_in_editor)
        controls.addWidget(self.editor_button)
        canvas = QVBoxLayout()
        root.addLayout(canvas, 1)
        toolbar = QHBoxLayout()
        self.view = MapView()
        self.view.color_picked.connect(self.pick_color)
        self.view.candidate_clicked.connect(self.remove_candidate)
        for name in ("walls", "doors", "lights"):
            check = QCheckBox(name.capitalize())
            check.setChecked(True)
            check.toggled.connect(lambda checked, name=name: self.view.show_layer(name, checked))
            toolbar.addWidget(check)
        toolbar.addStretch()
        remove = QCheckBox("Click to remove door / light")
        remove.toggled.connect(lambda checked: setattr(self.view, "removing", checked))
        toolbar.addWidget(remove)
        self.undo_button = QPushButton("Undo removal")
        self.undo_button.clicked.connect(self.undo_removal)
        self.undo_button.setEnabled(False)
        toolbar.addWidget(self.undo_button)
        fit = QPushButton("Fit map")
        fit.clicked.connect(self.view.fit_map)
        toolbar.addWidget(fit)
        canvas.addLayout(toolbar)
        canvas.addWidget(self.view, 1)
        self._update_swatches()

    def _number(self, low, high, value, decimals=0):
        widget = QDoubleSpinBox() if decimals else QSpinBox()
        if decimals:
            widget.setDecimals(decimals)
            widget.setSingleStep(0.1 if decimals == 1 else 0.05)
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self.settings_changed)
        return widget

    def _add_tab(self, label, layout):
        widget = QWidget()
        widget.setLayout(layout)
        self.tabs.addTab(widget, label)

    def _color_controls(self, target, layout):
        row = QHBoxLayout()
        swatch = QPushButton("Color…")
        swatch.clicked.connect(lambda: self.choose_color(target))
        pick = QPushButton("Pick on map")
        pick.clicked.connect(lambda: self.start_pick(target))
        row.addWidget(swatch)
        row.addWidget(pick)
        layout.addRow(row)
        return swatch

    def _update_swatches(self):
        for target in ("wall", "door"):
            rgb = tuple(getattr(self, target + "_color"))
            getattr(self, target + "_swatch").setStyleSheet(
                f"background-color: rgb{rgb}; color: {'black' if sum(rgb) > 380 else 'white'};")

    def choose_color(self, target):
        selected = QColorDialog.getColor(QColor(*getattr(self, target + "_color")), self)
        if selected.isValid():
            self._set_color(target, (selected.red(), selected.green(), selected.blue()))

    def _set_color(self, target, rgb):
        setattr(self, target + "_color", rgb)
        if target == "wall":
            self.wall_mode.setCurrentIndex(1)
        self._update_swatches()
        self.settings_changed()

    def start_pick(self, target):
        if self.image is not None:
            self.pick_target = target
            self.view.picking = True
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            self.status.setText(f"Click a {target} color in the map.")

    def pick_color(self, x, y):
        self._set_color(self.pick_target, tuple(int(v) for v in self.image[y, x, ::-1]))
        self.view.picking = False
        self.view.unsetCursor()
        self.pick_target = None

    def settings(self):
        return FastSettings(max_dimension=self.quality.currentData(), grid_size=self.grid.value(),
                            wall_mode=self.wall_mode.currentData(), wall_color=self.wall_color,
                            wall_tolerance=self.wall_tolerance.value(), min_wall_area=self.min_area.value(),
                            simplify=self.simplify.value(), doors=self.enable_doors.isChecked(),
                            door_color=self.door_color, door_hue_tolerance=self.hue.value(),
                            door_min_length=self.min_door.value(), door_max_length=self.max_door.value(),
                            door_rectangularity=self.rectangularity.value(), door_wall_support=self.support.isChecked(),
                            lights=self.enable_lights.isChecked(), light_threshold=self.brightness.value(),
                            light_min_area=self.min_light.value(), light_max_area=self.max_light.value(),
                            light_saturation=self.saturation.value(), light_range=self.light_range.value())

    def apply_settings(self, settings):
        """Use the same settings in GUI and headless runs."""
        settings.validate()
        quality_index = self.quality.findData(settings.max_dimension)
        if quality_index < 0:
            self.quality.addItem(f"Custom · {settings.max_dimension} px", settings.max_dimension)
            quality_index = self.quality.count() - 1
        self.quality.setCurrentIndex(quality_index)
        self.wall_mode.setCurrentIndex(self.wall_mode.findData(settings.wall_mode))
        self.wall_color, self.door_color = settings.wall_color, settings.door_color
        for widget, value in ((self.grid, settings.grid_size), (self.wall_tolerance, settings.wall_tolerance),
                              (self.min_area, settings.min_wall_area), (self.simplify, settings.simplify),
                              (self.hue, settings.door_hue_tolerance), (self.min_door, settings.door_min_length),
                              (self.max_door, settings.door_max_length), (self.rectangularity, settings.door_rectangularity),
                              (self.brightness, settings.light_threshold), (self.saturation, settings.light_saturation),
                              (self.min_light, settings.light_min_area), (self.max_light, settings.light_max_area),
                              (self.light_range, settings.light_range)):
            widget.setValue(value)
        self.enable_doors.setChecked(settings.doors)
        self.enable_lights.setChecked(settings.lights)
        self.support.setChecked(settings.door_wall_support)
        self._update_swatches()
        self.timer.stop()

    def settings_changed(self, *_):
        if not hasattr(self, "live"):
            return
        self.export_button.setEnabled(False)
        self.editor_button.setEnabled(False)
        if self.live.isChecked() and (self.image is not None or self.path):
            self.timer.start(200)
        elif self.result:
            self.status.setText("Settings changed. Click Detect all to update.")

    def open_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open map", "", "Maps (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.path = path
            self.run_detection()

    def run_detection(self):
        self.timer.stop()
        if self.closing or (self.image is None and not self.path):
            return
        try:
            settings = self.settings()
            settings.validate()
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.generation += 1
        self.export_button.setEnabled(False)
        self.editor_button.setEnabled(False)
        if self.worker is not None:
            self.pending = True
            self.worker.requestInterruption()
            return
        self.pending = False
        self.worker = DetectionWorker(self.image, self.source, self.path, settings, self.generation, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.detected)
        self.worker.failed.connect(self.detection_failed)
        self.worker.finished.connect(self.detection_finished)
        self.cancel_button.setEnabled(True)
        self.worker.start()

    def detected(self, payload):
        image, source, result, settings, generation = payload
        if self.closing or generation != self.generation:
            return
        if image is not self.image:
            self.view.set_image(image)
        self.image, self.source, self.result, self.result_settings = image, source, result, settings
        self.path = None
        self.undo_states.clear()
        self.undo_button.setEnabled(False)
        self.view.set_result(result)
        self.show_counts()
        current = settings == self.settings() and not self.timer.isActive()
        self.export_button.setEnabled(current and self.export_worker is None)
        self.editor_button.setEnabled(current and self.export_worker is None)

    def show_counts(self):
        result = self.result
        text = (f"{len(result.walls):,} wall segments · {len(result.doors)} doors · {len(result.lights)} lights\n"
                f"Detection: {result.timings['total']:.3f} s · {result.working_size[0]} × {result.working_size[1]}\n"
                "Green: walls · Cyan: doors · Yellow: lights")
        if result.warnings:
            text += "\n" + "\n".join(result.warnings)
        self.status.setText(text)

    def detection_failed(self, message):
        if not self.pending:
            self.status.setText("Detection failed: " + message)

    def detection_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.cancel_button.setEnabled(False)
        if self.closing:
            self._finish_close()
        elif self.pending:
            self.run_detection()

    def cancel_detection(self):
        self.timer.stop()
        self.pending = False
        self.generation += 1
        if self.worker:
            self.worker.requestInterruption()
        self.status.setText("Detection cancelled. Previous preview retained.")

    def remove_candidate(self, layer, index):
        if not self.result or self.worker or self.timer.isActive():
            return
        values = getattr(self.result, layer)
        self.undo_states.append((layer, index, values.pop(index)))
        self.view.set_result(self.result)
        self.undo_button.setEnabled(True)
        self.show_counts()

    def undo_removal(self):
        if not self.undo_states or self.worker:
            return
        layer, index, value = self.undo_states.pop()
        getattr(self.result, layer).insert(index, value)
        self.view.set_result(self.result)
        self.undo_button.setEnabled(bool(self.undo_states))
        self.show_counts()

    def export_map(self):
        if not self.result or self.export_worker:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export map", "map.dd2vtt", "Universal VTT (*.dd2vtt *.uvtt)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".dd2vtt", ".uvtt"):
            path += ".dd2vtt"
        self.export_button.setEnabled(False)
        self.editor_button.setEnabled(False)
        self.status.setText("Exporting original image, walls, doors, and lights…")
        self.export_worker = ExportWorker(path, self.image, self.source, self.result, self.result_settings, self)
        self.export_worker.completed.connect(lambda value: self.status.setText("Saved " + value))
        self.export_worker.failed.connect(lambda value: self.status.setText("Export failed: " + value))
        self.export_worker.finished.connect(self.export_finished)
        self.export_worker.start()

    def export_finished(self):
        self.export_worker.deleteLater()
        self.export_worker = None
        current = self.worker is None and not self.timer.isActive() and self.result_settings == self.settings()
        self.export_button.setEnabled(current)
        self.editor_button.setEnabled(current)
        if self.closing:
            self._finish_close()

    def use_in_editor(self):
        app = self.parent()
        if not app or not self.result or self.worker or self.export_worker:
            return
        app.image_processor.debounced_update.timer.stop()
        app.original_image = self.image
        app.current_image, app.scale_factor = app.image_processor.create_working_image(self.image)
        app.mask_layer = None
        app.bg_removed_image = None
        app.current_contours = []
        app.history.clear()
        app.image_processor.detection_cache.clear()
        app.uvtt_walls_preview = to_uvtt(self.result, self.result_settings.grid_size, preview=True)
        # The legacy exporter attaches image bytes only when Save is requested.
        app.fast_source_image = (self.image, self.source)
        app.current_lights = deepcopy(app.uvtt_walls_preview["lights"])
        app.uvtt_export_params = {"overlay_grid_size": self.result_settings.grid_size, "show_grid_overlay": False}
        app.uvtt_overlay_grid_size_spinbox.setValue(round(self.result_settings.grid_size))
        app.wall_edit_history = []
        app.uvtt_tool_btn.setChecked(True)
        app.on_tool_changed(app.uvtt_tool_btn)
        app.export_panel.save_wall_state_for_undo(force=True)
        app.export_panel.save_wall_state_for_undo(force=True)
        self.accept()

    def _finish_close(self):
        if self.worker is None and self.export_worker is None:
            super().reject()

    def reject(self):
        self.timer.stop()
        self.pending = False
        self.closing = True
        if self.worker:
            self.worker.requestInterruption()
        self._finish_close()

    def closeEvent(self, event):
        if self.worker or self.export_worker:
            self.reject()
            event.ignore()
        else:
            self.timer.stop()
            event.accept()
