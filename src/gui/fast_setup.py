"""Native desktop workspace for fast wall, door, and light setup."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from src.core.fast_pipeline import (
    DetectionCancelled, FastSettings, detect_map, export_image_bytes, load_map,
    to_uvtt, write_uvtt,
)
from src.core.detection_profiles import PRESETS, load_settings, preset_settings, save_settings
from src.core.wall_editing import cut_door_opening, incident_segments, move_vertex, nearest_wall, walls_in_rectangle


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
    wall_vertex_moved = pyqtSignal(object, object)
    wall_rectangle_deleted = pyqtSignal(object, object)
    segment_drawn = pyqtSignal(str, object, object)
    drawing_mode_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#14171b"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.bitmap = None
        self.layers = {"walls": [], "doors": [], "lights": [], "transparent": []}
        self.visibility = {name: True for name in self.layers}
        self.picking = False
        self.removing = False
        self.image_size = (0, 0)
        self.walls, self.doors = [], []
        self.editing_enabled = False
        self.drawing_mode = None
        self.draw_origin = self.draw_target = None
        self.drag_source = None
        self.drag_target = None
        self.drag_start = None
        self.drag_walls, self.drag_doors, self.snap_walls = [], [], []
        self.drag_preview = None
        self.snap_marker = None
        self.erase_origin = None
        self.erase_box = None
        self.setToolTip("Right-drag a wall to stretch its nearest end and snap to another wall.\n"
                        "Middle-drag a rectangle to delete walls it touches.\n"
                        "1: draw walls · 2: draw doors · press again to pan. Ctrl+Z: undo.")

    def set_image(self, image):
        self.cancel_wall_gesture()
        self.walls, self.doors = [], []
        self.editing_enabled = False
        self.update_cursor()
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
        self.cancel_wall_gesture()
        self.walls, self.doors = result.walls, result.doors
        for items in self.layers.values():
            for item in items:
                self.scene().removeItem(item)
        self.layers = {name: [] for name in self.layers}
        self.layers["walls"].append(self.scene().addPath(self.wall_path(result.walls), self.pen("#4ee19b", 1.6)))
        path = QPainterPath()
        for start, end in result.transparent_barriers:
            path.moveTo(*start)
            path.lineTo(*end)
        pen = self.pen("#d5a5e8", 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        self.layers["transparent"].append(self.scene().addPath(path, pen))
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
        if name in ("walls", "doors"):
            self.cancel_wall_gesture()
        self.visibility[name] = visible
        for item in self.layers[name]:
            item.setVisible(visible)
        if not visible and self.drawing_mode == name:
            self.set_drawing_mode(None)

    def has_wall_gesture(self):
        return self.drag_source is not None or self.erase_box is not None or self.draw_origin is not None

    def wheelEvent(self, event):
        if self.has_wall_gesture():
            event.accept()
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        zoom = self.transform().m11() * factor
        if 0.025 <= zoom <= 20:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if self.has_wall_gesture():
            event.accept()
            return
        if self.editing_enabled and self.visibility["walls"] and not self.picking:
            point = self.mapToScene(event.position().toPoint())
            xy = (point.x(), point.y())
            if event.button() == Qt.MouseButton.RightButton:
                hit = nearest_wall(self.walls, xy, 10 / self.transform().m11())
                if hit is not None:
                    segment = self.walls[hit[0]]
                    source = min(segment, key=lambda p: (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2)
                    self.drag_source = self.drag_target = tuple(source)
                    self.drag_start = event.position().toPoint()
                    wall_indices = set(incident_segments(self.walls, source))
                    door_indices = set(incident_segments(self.doors, source))
                    self.drag_walls = [wall for index, wall in enumerate(self.walls) if index in wall_indices]
                    self.drag_doors = [door for index, door in enumerate(self.doors) if index in door_indices]
                    self.snap_walls = [wall for index, wall in enumerate(self.walls) if index not in wall_indices]
                    self.layers["walls"][0].setPath(self.wall_path(self.snap_walls))
                    for index in door_indices:
                        self.layers["doors"][index].setVisible(False)
                    self.drag_preview = self.scene().addPath(QPainterPath(), self.pen("#ffd75e", 2.4))
                    self.snap_marker = self.scene().addEllipse(QRectF(), self.pen("#ffd75e", 2))
                    self.drag_preview.setZValue(10)
                    self.snap_marker.setZValue(11)
                    self.setFocus(Qt.FocusReason.MouseFocusReason)
                    self.setCursor(Qt.CursorShape.CrossCursor)
                    self.update_wall_drag(event.position().toPoint())
                event.accept()
                return
            if event.button() == Qt.MouseButton.MiddleButton:
                self.erase_origin = point
                self.drag_start = event.position().toPoint()
                self.erase_box = self.scene().addRect(QRectF(point, point), self.pen("#ff6565", 1.5), QColor(255, 80, 80, 35))
                self.erase_box.setZValue(10)
                self.setFocus(Qt.FocusReason.MouseFocusReason)
                self.setCursor(Qt.CursorShape.CrossCursor)
                event.accept()
                return
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
            if self.editing_enabled and self.drawing_mode and self.visibility[self.drawing_mode]:
                if 0 <= point.x() < self.image_size[0] and 0 <= point.y() < self.image_size[1]:
                    self.snap_walls = []
                    if self.visibility["walls"]:
                        self.snap_walls.extend(self.walls)
                    if self.visibility["doors"]:
                        self.snap_walls.extend(self.doors)
                    self.draw_origin, _ = self.snapped_point(event.position().toPoint(), self.snap_walls)
                    self.drag_start = event.position().toPoint()
                    color, width = ("#4ee19b", 2.4) if self.drawing_mode == "walls" else ("#35d4ff", 4)
                    self.drag_preview = self.scene().addPath(QPainterPath(), self.pen(color, width))
                    self.snap_marker = self.scene().addEllipse(QRectF(), self.pen("#ffd75e", 2))
                    self.drag_preview.setZValue(10)
                    self.snap_marker.setZValue(11)
                    self.setFocus(Qt.FocusReason.MouseFocusReason)
                    self.update_drawing(event.position().toPoint())
                event.accept()
                return
        super().mousePressEvent(event)

    @staticmethod
    def wall_path(segments):
        path = QPainterPath()
        for start, end in segments:
            path.moveTo(*start)
            path.lineTo(*end)
        return path

    def set_editing_enabled(self, enabled):
        self.editing_enabled = enabled
        if not enabled:
            self.cancel_wall_gesture()
        self.update_cursor()

    def set_drawing_mode(self, mode):
        if mode not in (None, "walls", "doors"):
            raise ValueError("Drawing mode must be walls, doors, or None")
        if mode == self.drawing_mode:
            return
        self.cancel_wall_gesture()
        self.drawing_mode = mode
        self.drawing_mode_changed.emit(mode)
        self.update_cursor()

    def update_cursor(self):
        if self.has_wall_gesture() or self.picking or (self.editing_enabled and self.drawing_mode):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def cancel_wall_gesture(self):
        if self.drag_source is not None:
            if self.layers["walls"]:
                self.layers["walls"][0].setPath(self.wall_path(self.walls))
            for item in self.layers["doors"]:
                item.setVisible(self.visibility["doors"])
        for item in (self.drag_preview, self.snap_marker, self.erase_box):
            if item is not None:
                self.scene().removeItem(item)
        self.drag_source = self.drag_target = self.drag_start = None
        self.draw_origin = self.draw_target = None
        self.drag_preview = self.snap_marker = self.erase_origin = self.erase_box = None
        self.drag_walls, self.drag_doors, self.snap_walls = [], [], []
        self.update_cursor()

    def snapped_point(self, position, segments):
        point = self.mapToScene(position)
        target = (min(max(point.x(), 0), self.image_size[0] - 1),
                  min(max(point.y(), 0), self.image_size[1] - 1))
        hit = nearest_wall(segments, target, 10 / self.transform().m11())
        return (hit[1], True) if hit is not None else (target, False)

    def update_snap_marker(self, target, snapped):
        radius = 5 / self.transform().m11()
        self.snap_marker.setRect(target[0] - radius, target[1] - radius, radius * 2, radius * 2)
        self.snap_marker.setPen(self.pen("#4ee19b" if snapped else "#ffd75e", 2))

    def update_wall_drag(self, position):
        if (position - self.drag_start).manhattanLength() < 3:
            target, snapped = self.drag_source, False
        else:
            target, snapped = self.snapped_point(position, self.snap_walls)
        self.drag_target = target
        segments = move_vertex(self.drag_walls, self.drag_source, target)
        if self.visibility["doors"]:
            segments += move_vertex(self.drag_doors, self.drag_source, target)
        self.drag_preview.setPath(self.wall_path(segments))
        self.update_snap_marker(target, snapped)

    def update_drawing(self, position):
        if (position - self.drag_start).manhattanLength() < 3:
            target, snapped = self.draw_origin, False
        else:
            target, snapped = self.snapped_point(position, self.snap_walls)
        self.draw_target = target
        self.drag_preview.setPath(self.wall_path([(self.draw_origin, target)]))
        self.update_snap_marker(target, snapped)

    def mouseMoveEvent(self, event):
        if self.drag_source is not None:
            self.update_wall_drag(event.position().toPoint())
            event.accept()
        elif self.erase_box is not None:
            self.erase_box.setRect(QRectF(self.erase_origin, self.mapToScene(event.position().toPoint())).normalized())
            event.accept()
        elif self.draw_origin is not None:
            self.update_drawing(event.position().toPoint())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.drag_source is not None:
            self.update_wall_drag(event.position().toPoint())
            source, target = self.drag_source, self.drag_target
            self.cancel_wall_gesture()
            if source != target:
                self.wall_vertex_moved.emit(source, target)
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton and self.erase_box is not None:
            rect = QRectF(self.erase_origin, self.mapToScene(event.position().toPoint())).normalized()
            moved = (event.position().toPoint() - self.drag_start).manhattanLength() >= 3
            self.cancel_wall_gesture()
            if moved and rect.width() > 0 and rect.height() > 0:
                self.wall_rectangle_deleted.emit((rect.left(), rect.top()), (rect.right(), rect.bottom()))
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self.draw_origin is not None:
            self.update_drawing(event.position().toPoint())
            source, target, layer = self.draw_origin, self.draw_target, self.drawing_mode
            self.cancel_wall_gesture()
            if source != target:
                self.segment_drawn.emit(layer, source, target)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_1, Qt.Key.Key_2) and event.modifiers() in (
                Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
            if self.editing_enabled and not event.isAutoRepeat():
                mode = "walls" if event.key() == Qt.Key.Key_1 else "doors"
                self.set_drawing_mode(None if self.drawing_mode == mode else mode)
            event.accept()
        elif event.key() == Qt.Key.Key_Escape and (self.has_wall_gesture() or self.drawing_mode):
            if self.has_wall_gesture():
                self.cancel_wall_gesture()
            else:
                self.set_drawing_mode(None)
            event.accept()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self.cancel_wall_gesture()
        super().focusOutEvent(event)


class FastSetupDialog(QDialog):
    """Single-flight workers, with newer requests replacing obsolete work."""

    def __init__(self, parent=None, image=None, source=None, path=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Wall • Fast Setup")
        self.resize(1440, 960)
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
        self.wall_color = FastSettings().wall_color
        self.door_color = FastSettings().door_color
        self.void_color = FastSettings().void_color
        self.light_color = FastSettings().light_color
        self.applying_settings = False
        self.undo_states = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.run_detection)
        self._build_ui()
        self.apply_settings(FastSettings())
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
        caption = QLabel("Scroll to zoom. Left-drag to pan.\nRight-drag to stretch a wall.\n"
                         "Middle-drag a box to delete walls.\n1: draw walls · 2: draw doors.")
        controls.addWidget(caption)
        open_button = QPushButton("Open map…")
        open_button.clicked.connect(self.open_map)
        controls.addWidget(open_button)
        self.preset = QComboBox()
        for key, label in PRESETS.items():
            self.preset.addItem(label, key)
        self.preset.addItem("Custom / imported settings", None)
        self.preset.currentIndexChanged.connect(self.select_preset)
        controls.addWidget(self.preset)
        profile_row = QHBoxLayout()
        import_button = QPushButton("Import JSON…")
        import_button.setToolTip("Import a map-style palette schema or saved detector settings")
        import_button.clicked.connect(self.import_settings)
        save_button = QPushButton("Save settings…")
        save_button.clicked.connect(self.save_current_settings)
        profile_row.addWidget(import_button)
        profile_row.addWidget(save_button)
        controls.addLayout(profile_row)
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
        self.wall_tolerance = self._number(0, 255, 24)
        walls.addRow("Color tolerance", self.wall_tolerance)
        self.min_area = self._number(0, 5000, 1000, decimals=1)
        walls.addRow("Minimum area", self.min_area)
        self.simplify = self._number(0, 10, 1.25, decimals=2)
        walls.addRow("Simplify", self.simplify)
        self.geometry = QComboBox()
        self.geometry.addItem("Outlines", "outline")
        self.geometry.addItem("Centerlines", "centerline")
        self.geometry.currentIndexChanged.connect(self.settings_changed)
        walls.addRow("Geometry", self.geometry)
        self.gap_close = self._number(0, 10, 0, decimals=1)
        self.gap_close.setToolTip("0 preserves openings. Increase only to repair artwork seams.")
        walls.addRow("Repair gaps", self.gap_close)
        self._add_tab("Walls", walls)
        doors = QFormLayout()
        self.enable_doors = QCheckBox("Auto-detect doors")
        self.enable_doors.setChecked(True)
        self.enable_doors.toggled.connect(self.settings_changed)
        doors.addRow(self.enable_doors)
        self.door_swatch = self._color_controls("door", doors)
        self.door_match = QComboBox()
        self.door_match.addItem("Hue + sampled brightness", "hue")
        self.door_match.addItem("RGB palette", "rgb")
        self.door_match.currentIndexChanged.connect(self.settings_changed)
        doors.addRow("Color method", self.door_match)
        self.door_tolerance = self._number(0, 255, 24)
        doors.addRow("RGB tolerance", self.door_tolerance)
        self.hue = self._number(0, 90, 16)
        doors.addRow("Hue tolerance", self.hue)
        self.min_door = self._number(2, 400, 12, decimals=1)
        self.max_door = self._number(2, 800, 60, decimals=1)
        doors.addRow("Min length", self.min_door)
        doors.addRow("Max length", self.max_door)
        self.door_gap = self._number(0, 10, 2, decimals=1)
        doors.addRow("Repair seams", self.door_gap)
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
        self.light_mode = QComboBox()
        for label, value in (("Warm sources", "warm"), ("Cool sources", "cool"),
                             ("Sampled hue", "color"), ("All bright regions", "brightness")):
            self.light_mode.addItem(label, value)
        self.light_mode.currentIndexChanged.connect(self.settings_changed)
        lights.addRow("Method", self.light_mode)
        self.light_swatch = self._color_controls("light", lights)
        self.brightness = self._number(0, 255, 190)
        self.saturation = self._number(0, 255, 60)
        self.min_light = self._number(1, 5000, 3, decimals=1)
        self.max_light = self._number(1, 10000, 180, decimals=1)
        self.light_core = self._number(0, 1, 0.06, decimals=2)
        self.light_fill = self._number(0, 1, 0.25, decimals=2)
        self.light_range = self._number(0.1, 100, 6, decimals=1)
        lights.addRow("Brightness", self.brightness)
        lights.addRow("Min saturation", self.saturation)
        lights.addRow("Min area", self.min_light)
        lights.addRow("Max area", self.max_light)
        lights.addRow("Bright fraction", self.light_core)
        lights.addRow("Min fill ratio", self.light_fill)
        self.light_core.setToolTip("Fraction of the colored source that must exceed the brightness threshold")
        self.light_fill.setToolTip("Colored area divided by its bounding box; rejects sparse markings")
        lights.addRow("Range (squares)", self.light_range)
        self._add_tab("Lights", lights)
        ml = QFormLayout()
        explanation = QLabel("Local models are experimental. Review the preview before export.")
        explanation.setWordWrap(True)
        ml.addRow(explanation)
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.model_path.setPlaceholderText("No model · palette detection")
        ml.addRow(self.model_path)
        model_buttons = QHBoxLayout()
        choose_model = QPushButton("Choose model…")
        choose_model.clicked.connect(self.choose_model)
        clear_model = QPushButton("Use palette")
        clear_model.clicked.connect(self.clear_model)
        model_buttons.addWidget(choose_model)
        model_buttons.addWidget(clear_model)
        ml.addRow(model_buttons)
        self.ml_device = QComboBox()
        for name, device in (("Automatic", "auto"), ("CPU", "cpu"), ("CUDA GPU", "cuda")):
            self.ml_device.addItem(name, device)
        self.ml_device.currentIndexChanged.connect(self.settings_changed)
        ml.addRow("Device", self.ml_device)
        self.ml_wall = self._number(.5, 1, .9, decimals=2)
        self.ml_door = self._number(.5, 1, .9, decimals=2)
        self.ml_light = self._number(.5, 1, .9, decimals=2)
        self.ml_light.setToolTip("Light-pixel threshold for models without light-center prediction.")
        self.ml_light_center = self._number(.05, 1, .35, decimals=2)
        self.ml_light_center.setToolTip("For models that find whole light sources. Higher scores accept fewer sources; this is not a measured accuracy percentage.")
        ml.addRow("Wall confidence", self.ml_wall)
        ml.addRow("Door confidence", self.ml_door)
        ml.addRow("Light confidence", self.ml_light)
        ml.addRow("Light center score", self.ml_light_center)
        self.ml_recover_jambs = QCheckBox("Recover shaded door jambs")
        self.ml_recover_jambs.setToolTip("Use connected, lower-confidence wall predictions only beside supported doors. Review the result.")
        self.ml_recover_jambs.toggled.connect(self.settings_changed)
        ml.addRow(self.ml_recover_jambs)
        ml_note = QLabel("Point models use Light center score. Other models use Light confidence and the Lights tab's area limits.")
        ml_note.setWordWrap(True)
        ml.addRow(ml_note)
        self._add_tab("ML", ml)
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
        self.view.wall_vertex_moved.connect(self.stretch_wall)
        self.view.wall_rectangle_deleted.connect(self.delete_walls_in_rectangle)
        self.view.segment_drawn.connect(self.add_segment)
        self.view.drawing_mode_changed.connect(self.drawing_mode_changed)
        self.layer_checks = {}
        for name in ("walls", "doors", "lights", "transparent"):
            check = QCheckBox("See-through (not exported)" if name == "transparent" else name.capitalize())
            check.setChecked(True)
            check.toggled.connect(lambda checked, name=name: self.view.show_layer(name, checked))
            self.layer_checks[name] = check
            toolbar.addWidget(check)
        toolbar.addStretch()
        self.remove_check = QCheckBox("Click to remove door / light")
        self.remove_check.toggled.connect(self.set_removing)
        toolbar.addWidget(self.remove_check)
        self.undo_button = QPushButton("Undo edit")
        self.undo_button.setToolTip("Undo the last drawing, wall edit, or door/light removal (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_removal)
        self.undo_button.setEnabled(False)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo_removal)
        toolbar.addWidget(self.undo_button)
        fit = QPushButton("Fit map")
        fit.clicked.connect(self.view.fit_map)
        toolbar.addWidget(fit)
        canvas.addLayout(toolbar)
        drawing_toolbar = QHBoxLayout()
        self.draw_buttons = {}
        for layer, label in (("walls", "1 · Draw wall"), ("doors", "2 · Draw door")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip("Left-drag to draw. Press this number while the map is focused to toggle the tool.")
            button.clicked.connect(lambda checked, layer=layer: self.view.set_drawing_mode(layer if checked else None))
            self.draw_buttons[layer] = button
            drawing_toolbar.addWidget(button)
        self.drawing_hint = QLabel("Left-drag: pan · 1 / 2: toggle drawing")
        drawing_toolbar.addWidget(self.drawing_hint)
        drawing_toolbar.addStretch()
        canvas.addLayout(drawing_toolbar)
        canvas.addWidget(self.view, 1)
        self._update_swatches()

    def drawing_mode_changed(self, mode):
        if mode:
            self.pick_target = None
            self.view.picking = False
            self.remove_check.setChecked(False)
            self.layer_checks[mode].setChecked(True)
        for layer, button in self.draw_buttons.items():
            button.setChecked(mode == layer)
        self.drawing_hint.setText(f"Left-drag: draw {'walls' if mode == 'walls' else 'doors'} · Esc: cancel / pan"
                                  if mode else "Left-drag: pan · 1 / 2: toggle drawing")
        self.view.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_removing(self, enabled):
        self.view.cancel_wall_gesture()
        if enabled:
            self.view.set_drawing_mode(None)
            self.pick_target = None
            self.view.picking = False
        self.view.removing = enabled
        self.view.update_cursor()

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
        for target in ("wall", "door", "light"):
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
        elif target == "light":
            self.light_mode.setCurrentIndex(self.light_mode.findData("color"))
        self._update_swatches()
        self.settings_changed()

    def start_pick(self, target):
        if self.image is not None:
            self.view.set_drawing_mode(None)
            self.view.cancel_wall_gesture()
            self.remove_check.setChecked(False)
            self.pick_target = target
            self.view.picking = True
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            self.status.setText(f"Click a {target} color in the map.")

    def pick_color(self, x, y):
        self._set_color(self.pick_target, tuple(int(v) for v in self.image[y, x, ::-1]))
        self.view.picking = False
        self.view.update_cursor()
        self.pick_target = None

    def select_preset(self):
        name = self.preset.currentData()
        if name is None or self.applying_settings:
            return
        settings = replace(preset_settings(name), max_dimension=self.quality.currentData(), grid_size=self.grid.value())
        self.apply_settings(settings)
        self.settings_changed()
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData(name))
        self.preset.blockSignals(False)

    def import_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import map schema or detector settings", "", "JSON (*.json)")
        if path:
            try:
                self.apply_settings(load_settings(path, self.settings()))
                self.settings_changed()
            except (OSError, ValueError) as error:
                self.status.setText("Could not import settings: " + str(error))

    def save_current_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save detector settings", "map-settings.json", "JSON (*.json)")
        if path:
            try:
                save_settings(path, self.settings())
                self.status.setText("Saved detector settings: " + path)
            except (OSError, ValueError) as error:
                self.status.setText("Could not save settings: " + str(error))

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose local Auto-Wall model", "", "Auto-Wall models (*.safetensors)")
        if path:
            self.model_path.setText(path)
            self.model_path.setToolTip(path)
            self.settings_changed()

    def clear_model(self):
        self.model_path.clear()
        self.model_path.setToolTip("")
        self.settings_changed()

    def settings(self):
        return FastSettings(max_dimension=self.quality.currentData(), grid_size=self.grid.value(),
                            wall_mode=self.wall_mode.currentData(), wall_color=self.wall_color,
                            wall_tolerance=self.wall_tolerance.value(), min_wall_area=self.min_area.value(),
                            simplify=self.simplify.value(), wall_gap_close=self.gap_close.value(),
                            wall_geometry=self.geometry.currentData(), void_color=self.void_color,
                            doors=self.enable_doors.isChecked(), door_match=self.door_match.currentData(),
                            door_color_tolerance=self.door_tolerance.value(),
                            door_color=self.door_color, door_hue_tolerance=self.hue.value(),
                            door_min_length=self.min_door.value(), door_max_length=self.max_door.value(),
                            door_gap_close=self.door_gap.value(),
                            door_rectangularity=self.rectangularity.value(), door_wall_support=self.support.isChecked(),
                            lights=self.enable_lights.isChecked(), light_mode=self.light_mode.currentData(),
                            light_color=self.light_color, light_threshold=self.brightness.value(),
                            light_min_area=self.min_light.value(), light_max_area=self.max_light.value(),
                            light_core_fraction=self.light_core.value(), light_min_fill=self.light_fill.value(),
                            light_saturation=self.saturation.value(), light_range=self.light_range.value(),
                            model_path=self.model_path.text(), ml_device=self.ml_device.currentData(),
                            ml_wall_confidence=self.ml_wall.value(), ml_door_confidence=self.ml_door.value(),
                            ml_light_confidence=self.ml_light.value(), ml_light_center_threshold=self.ml_light_center.value(),
                            ml_recover_jambs=self.ml_recover_jambs.isChecked())

    def apply_settings(self, settings):
        """Use the same settings in GUI and headless runs."""
        settings.validate()
        self.applying_settings = True
        quality_index = self.quality.findData(settings.max_dimension)
        if quality_index < 0:
            self.quality.addItem(f"Custom · {settings.max_dimension} px", settings.max_dimension)
            quality_index = self.quality.count() - 1
        self.quality.setCurrentIndex(quality_index)
        self.wall_mode.setCurrentIndex(self.wall_mode.findData(settings.wall_mode))
        self.wall_color, self.door_color = settings.wall_color, settings.door_color
        self.void_color = settings.void_color
        self.light_color = settings.light_color
        self.geometry.setCurrentIndex(self.geometry.findData(settings.wall_geometry))
        self.door_match.setCurrentIndex(self.door_match.findData(settings.door_match))
        self.light_mode.setCurrentIndex(self.light_mode.findData(settings.light_mode))
        self.model_path.setText(settings.model_path)
        self.model_path.setToolTip(settings.model_path)
        self.ml_device.setCurrentIndex(self.ml_device.findData(settings.ml_device))
        for widget, value in ((self.grid, settings.grid_size), (self.wall_tolerance, settings.wall_tolerance),
                              (self.min_area, settings.min_wall_area), (self.simplify, settings.simplify),
                              (self.gap_close, settings.wall_gap_close), (self.door_tolerance, settings.door_color_tolerance),
                              (self.door_gap, settings.door_gap_close),
                              (self.hue, settings.door_hue_tolerance), (self.min_door, settings.door_min_length),
                              (self.max_door, settings.door_max_length), (self.rectangularity, settings.door_rectangularity),
                              (self.brightness, settings.light_threshold), (self.saturation, settings.light_saturation),
                              (self.min_light, settings.light_min_area), (self.max_light, settings.light_max_area),
                              (self.light_core, settings.light_core_fraction), (self.light_fill, settings.light_min_fill),
                              (self.light_range, settings.light_range), (self.ml_wall, settings.ml_wall_confidence),
                              (self.ml_door, settings.ml_door_confidence), (self.ml_light, settings.ml_light_confidence),
                              (self.ml_light_center, settings.ml_light_center_threshold)):
            # Imported profiles may exceed a control's convenient initial range.
            # Expand it instead of silently changing a valid detector setting.
            widget.setRange(min(widget.minimum(), value), max(widget.maximum(), value))
            if isinstance(widget, QDoubleSpinBox):
                decimals = len(f"{value:.10f}".rstrip("0").split(".")[1])
                widget.setDecimals(max(widget.decimals(), decimals))
            widget.setValue(value)
        self.enable_doors.setChecked(settings.doors)
        self.enable_lights.setChecked(settings.lights)
        self.support.setChecked(settings.door_wall_support)
        self.ml_recover_jambs.setChecked(settings.ml_recover_jambs)
        self._update_swatches()
        self.timer.stop()
        match = next((name for name in PRESETS if preset_settings(name) == settings), None)
        self.preset.setCurrentIndex(self.preset.findData(match))
        self.applying_settings = False
        self.update_actions()

    def can_edit(self):
        return (self.result is not None and self.path is None and not self.closing
                and self.worker is None and self.export_worker is None
                and not self.timer.isActive() and self.result_settings == self.settings())

    def update_actions(self):
        current = self.can_edit()
        self.view.set_editing_enabled(current)
        self.export_button.setEnabled(current)
        self.editor_button.setEnabled(current)
        self.undo_button.setEnabled(current and bool(self.undo_states))
        for button in self.draw_buttons.values():
            button.setEnabled(current)

    def settings_changed(self, *_):
        if not hasattr(self, "live") or self.applying_settings:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData(None))
        self.preset.blockSignals(False)
        if self.live.isChecked() and (self.image is not None or self.path):
            self.timer.start(200)
        elif self.result:
            self.status.setText("Settings changed. Click Detect all to update.")
        self.update_actions()

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
        self.view.set_editing_enabled(False)
        self.undo_button.setEnabled(False)
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
        self.update_actions()

    def detected(self, payload):
        image, source, result, settings, generation = payload
        if self.closing or generation != self.generation:
            return
        if image is not self.image:
            self.view.set_image(image)
        self.image, self.source, self.result, self.result_settings = image, source, result, settings
        self.path = None
        self.undo_states.clear()
        self.view.set_result(result)
        self.show_counts()
        self.update_actions()

    def show_counts(self):
        result = self.result
        text = (f"{len(result.walls):,} wall segments · {len(result.doors)} doors · {len(result.lights)} lights\n"
                f"Detection: {result.timings['total']:.3f} s · {result.working_size[0]} × {result.working_size[1]}\n"
                "Green: walls · Cyan: doors · Yellow: lights")
        if result.detection_backend.startswith("torch-"):
            text += "\nML device: " + ("CUDA GPU" if result.detection_backend == "torch-cuda" else "CPU")
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
        else:
            self.update_actions()

    def cancel_detection(self):
        self.timer.stop()
        self.pending = False
        self.generation += 1
        if self.worker:
            self.worker.requestInterruption()
        self.status.setText("Detection cancelled. Previous preview retained.")
        self.update_actions()

    def apply_edit(self, **layers):
        if not self.can_edit():
            return
        self.undo_states.append({name: getattr(self.result, name) for name in layers})
        for name, values in layers.items():
            setattr(self.result, name, values)
        self.view.set_result(self.result)
        self.update_actions()
        self.show_counts()

    def stretch_wall(self, source, target):
        if not self.can_edit() or source == target:
            return
        # Shared corners and door jambs move together rather than leaving a crack.
        self.apply_edit(walls=move_vertex(self.result.walls, source, target),
                        doors=move_vertex(self.result.doors, source, target))

    def add_segment(self, layer, source, target):
        if layer not in ("walls", "doors") or not self.can_edit() or source == target:
            return
        values = getattr(self.result, layer)
        if (source, target) in values or (target, source) in values:
            return
        changes = {layer: [*values, (source, target)]}
        if layer == "doors":
            walls = cut_door_opening(self.result.walls, (source, target))
            if walls != self.result.walls:
                changes["walls"] = walls
        self.apply_edit(**changes)

    def delete_walls_in_rectangle(self, first, second):
        if not self.can_edit():
            return
        selected = set(walls_in_rectangle(self.result.walls, first, second))
        if selected:
            self.apply_edit(walls=[wall for index, wall in enumerate(self.result.walls) if index not in selected])

    def remove_candidate(self, layer, index):
        if not self.can_edit():
            return
        values = list(getattr(self.result, layer))
        values.pop(index)
        self.apply_edit(**{layer: values})

    def undo_removal(self):
        if not self.undo_states or not self.can_edit():
            return
        for layer, values in self.undo_states.pop().items():
            setattr(self.result, layer, values)
        self.view.set_result(self.result)
        self.update_actions()
        self.show_counts()

    def export_map(self):
        if not self.can_edit():
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
        self.update_actions()

    def export_finished(self):
        self.export_worker.deleteLater()
        self.export_worker = None
        self.update_actions()
        if self.closing:
            self._finish_close()

    def use_in_editor(self):
        app = self.parent()
        if not app or not self.can_edit():
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
        self.update_actions()
        if self.worker:
            self.worker.requestInterruption()
        self._finish_close()

    def closeEvent(self, event):
        if self.worker or self.export_worker:
            self.reject()
            event.ignore()
        else:
            self.timer.stop()
            self.view.set_editing_enabled(False)
            event.accept()
