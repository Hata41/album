import sys
import os
import random
import json
import copy
from pathlib import Path
from typing import List, Dict, Optional, Set

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem, 
                             QGraphicsPixmapItem, QGraphicsTextItem, QFileDialog, QLabel, QProgressBar,
                             QSplitter, QMessageBox, QFrame, QComboBox, QSpinBox, QLineEdit, QCheckBox, QMenu, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QImage, QPainter, QColor, QPen, QIcon, QTextDocument

import sa_advanced as sa
from PIL import Image, ImageOps

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def serialize_tree(node: Optional[sa.Node]) -> Optional[Dict]:
    if node is None:
        return None
    return {
        "dir": node.dir,
        "t": node.t,
        "leaf_id": node.leaf_id,
        "locked": node.locked,
        "left": serialize_tree(node.left),
        "right": serialize_tree(node.right)
    }

def deserialize_tree(data: Optional[Dict]) -> Optional[sa.Node]:
    if data is None:
        return None
    node = sa.Node(
        dir=data["dir"],
        t=data["t"],
        leaf_id=data["leaf_id"],
        locked=data.get("locked", False)
    )
    node.left = deserialize_tree(data["left"])
    node.right = deserialize_tree(data["right"])
    return node

class ImageListEntry(QWidget):
    def __init__(self, pixmap, filename, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.thumb = QLabel()
        self.thumb.setPixmap(pixmap)
        self.thumb.setFixedSize(60, 60)
        self.thumb.setScaledContents(True)
        # Allow clicks to pass through to the list item for selection/dragging
        self.thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.thumb)
        
        self.name_label = QLabel(filename)
        self.name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.name_label, 1)
        
        self.pool_cb = QCheckBox("Pool")
        self.pool_cb.setChecked(True)
        layout.addWidget(self.pool_cb)
        
        self.mandatory_cb = QCheckBox("Mandatory")
        layout.addWidget(self.mandatory_cb)
        
        self.pool_cb.toggled.connect(self.on_pool_toggled)
        
    def on_pool_toggled(self, checked):
        self.mandatory_cb.setEnabled(checked)
        if not checked:
            self.mandatory_cb.setChecked(False)

class OptimizationThread(QThread):
    progress = pyqtSignal(int, int)
    finished_optim = pyqtSignal(list, list)
    
    def __init__(self, chunks, swap_pool, pages_locks, all_prefs, image_paths, page_W, page_H, title, pages_roots, steps, mode, forced_aspects, show_page_numbers):
        super().__init__()
        self.chunks = chunks
        self.swap_pool = swap_pool.copy()
        self.pages_locks = pages_locks
        self.all_prefs = all_prefs
        self.image_paths = image_paths
        self.page_W = page_W
        self.page_H = page_H
        self.title = title
        self.pages_roots = pages_roots
        self.steps = steps
        self.mode = mode
        self.forced_aspects = forced_aspects
        self.show_page_numbers = show_page_numbers
        
    def run(self):
        if self.mode == "es":
            results, final_pool, energy_history = sa.optimize_es(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title,
                forced_aspects=self.forced_aspects,
                show_page_numbers=self.show_page_numbers
            )
        elif self.mode == "linear":
            new_roots, results, final_pool, energy_history = sa.optimize_linear_partition(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title,
                show_page_numbers=self.show_page_numbers
            )
            # Update roots in place
            for i in range(len(self.pages_roots)):
                self.pages_roots[i] = new_roots[i]
        else:
            results, final_pool, energy_history = sa.anneal_global(
                roots=self.pages_roots,
                page_W=self.page_W,
                page_H=self.page_H,
                all_images=self.image_paths,
                all_prefs=self.all_prefs,
                initial_perms=self.chunks,
                swap_pool=self.swap_pool,
                locked_leaves=self.pages_locks,
                steps=self.steps,
                progress_callback=self.emit_progress,
                title=self.title,
                forced_aspects=self.forced_aspects,
                show_page_numbers=self.show_page_numbers
            )
        self.finished_optim.emit(results, energy_history)
        
    def emit_progress(self, step, total):
        self.progress.emit(step, total)

class ExportThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, output_path, pages_roots, pages_perms, image_paths, page_titles, image_crop_states, show_labels, gap_px, label_bold, label_size_ratio, show_page_numbers):
        super().__init__()
        self.output_path = output_path
        self.pages_roots = pages_roots
        self.pages_perms = pages_perms
        self.image_paths = image_paths
        self.page_titles = page_titles
        self.image_crop_states = image_crop_states
        self.show_labels = show_labels
        self.gap_px = gap_px
        self.label_bold = label_bold
        self.label_size_ratio = label_size_ratio
        self.show_page_numbers = show_page_numbers
        self.export_W = 2480
        self.export_H = 3508
        
    def run(self):
        try:
            pdf_pages = []
            total = len(self.pages_roots)
            for i, (root, perm) in enumerate(zip(self.pages_roots, self.pages_perms)):
                if not root or not perm:
                    continue
                
                title = self.page_titles[i] if i < len(self.page_titles) else "Page"
                page_img = sa.render_page(
                    root=root,
                    page_W=self.export_W,
                    page_H=self.export_H,
                    images=self.image_paths,
                    perm=perm,
                    page_margin_px=120,
                    gap_px=self.gap_px,
                    title=title,
                    crop_states=self.image_crop_states,
                    show_labels=self.show_labels,
                    label_bold=self.label_bold,
                    label_size_ratio=self.label_size_ratio,
                    show_page_numbers=self.show_page_numbers,
                    page_num=i + 1
                )
                pdf_pages.append(page_img)
                self.progress.emit(i + 1, total)
            
            if pdf_pages:
                pdf_pages[0].save(
                    self.output_path, "PDF", resolution=300.0,
                    save_all=True, append_images=pdf_pages[1:]
                )
                self.finished.emit(True, f"Successfully saved to {self.output_path}")
            else:
                self.finished.emit(False, "No pages to export.")
        except Exception as e:
            self.finished.emit(False, str(e))

class LeafItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, page_idx, leaf_id, parent_gui):
        super().__init__(x, y, w, h)
        self.page_idx = page_idx
        self.leaf_id = leaf_id
        self.parent_gui = parent_gui
        self.setAcceptDrops(True)
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setBrush(QColor(255, 255, 255))
        
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        # Lock icon (simple red border or overlay for now)
        self.is_locked = False

    def set_locked(self, locked: bool):
        self.is_locked = locked
        self.update()
        
    def contextMenuEvent(self, event):
        if self.page_idx >= len(self.parent_gui.pages_perms):
            return
        perm = self.parent_gui.pages_perms[self.page_idx]
        if self.leaf_id >= len(perm):
            return
        img_idx = perm[self.leaf_id]
        
        menu = QMenu()
        is_cropped = self.parent_gui.image_crop_states.get(img_idx, True)
        
        action_text = "Fit Entire Image" if is_cropped else "Crop to Fill"
        toggle_action = menu.addAction(action_text)
        
        is_forced = img_idx in self.parent_gui.forced_aspect_ratios
        lock_aspect_action = menu.addAction("Lock Aspect Ratio")
        lock_aspect_action.setCheckable(True)
        lock_aspect_action.setChecked(is_forced)
        
        action = menu.exec(event.screenPos())
        if action == toggle_action:
            self.parent_gui.toggle_crop_state(img_idx)
        elif action == lock_aspect_action:
            self.parent_gui.toggle_aspect_lock(img_idx)

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.is_locked:
            painter.setPen(QPen(Qt.GlobalColor.red, 4))
            painter.drawRect(self.rect())
            
    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-image-idx"):
            data = event.mimeData().data("application/x-image-idx")
            img_idx = int(data.data().decode())
            self.parent_gui.handle_drop(self.page_idx, self.leaf_id, img_idx)
            event.accept()
        else:
            event.ignore()

class AlbumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Esthetic Album Designer")
        self.resize(1200, 800)
        
        # State
        self.image_paths: List[Path] = []
        self.current_folder: Optional[Path] = None
        self.image_metadata: List[sa.ImageMetadata] = []
        self.image_crop_states: Dict[int, bool] = {} # img_idx -> bool (True=Crop, False=Fit)
        self.forced_aspect_ratios: Set[int] = set()
        self.show_labels = True
        self.label_bold = False
        self.label_size_ratio = 0.5
        self.image_gap = 10
        self.show_page_numbers = False
        self.page_titles: List[str] = []
        self.pages_roots: List[Optional[sa.Node]] = []
        self.pages_perms: List[List[int]] = [] # Each is Maps leaf_id -> image_idx
        self.pages_locks: List[Dict[int, int]] = [] # Each is leaf_id -> image_idx
        self.target_leaf_count: Optional[int] = None
        self.all_prefs: List[float] = []
        self.current_page_idx = 0
        self.view_mode = "single" # "single" or "grid"
        
        self.page_W = 1000  # Internal logic size
        self.page_H = 1414  # ~A4 Aspect
        
        self.snapshots = []

        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Left: Images
        left_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_images_dialog)
        left_layout.addWidget(self.load_btn)
        
        # History / Snapshots
        snapshot_group = QGroupBox("History / Snapshots")
        snapshot_layout = QVBoxLayout()
        
        self.take_snapshot_btn = QPushButton("Take Snapshot")
        self.take_snapshot_btn.clicked.connect(self.take_snapshot)
        snapshot_layout.addWidget(self.take_snapshot_btn)
        
        self.snapshot_combo = QComboBox()
        snapshot_layout.addWidget(self.snapshot_combo)
        
        snapshot_btns = QHBoxLayout()
        self.restore_snapshot_btn = QPushButton("Restore")
        self.restore_snapshot_btn.clicked.connect(self.restore_snapshot)
        snapshot_btns.addWidget(self.restore_snapshot_btn)
        
        self.delete_snapshot_btn = QPushButton("Delete")
        self.delete_snapshot_btn.clicked.connect(self.delete_snapshot)
        snapshot_btns.addWidget(self.delete_snapshot_btn)
        
        snapshot_layout.addLayout(snapshot_btns)
        
        snapshot_group.setLayout(snapshot_layout)
        left_layout.addWidget(snapshot_group)
        
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_images)
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_none_images)
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.select_none_btn)
        left_layout.addLayout(select_layout)
        
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setDragEnabled(True)
        self.image_list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        left_layout.addWidget(self.image_list)
        
        # Center: Canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setBackgroundBrush(QColor(50, 50, 50))
        
        # Right: Controls
        right_layout = QVBoxLayout()

        # Statistics Dashboard
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stat_total_images = QLabel("Total Images: 0")
        self.stat_album_capacity = QLabel("Album Capacity: 0")
        self.stat_mandatory = QLabel("Mandatory: 0")
        self.stat_pool = QLabel("Pool: 0")
        stats_layout.addWidget(self.stat_total_images)
        stats_layout.addWidget(self.stat_album_capacity)
        stats_layout.addWidget(self.stat_mandatory)
        stats_layout.addWidget(self.stat_pool)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # Page Configuration
        config_row = QHBoxLayout()
        
        num_pages_label = QLabel("Number of Pages")
        self.num_pages_spin = QSpinBox()
        self.num_pages_spin.setMinimum(1)
        self.num_pages_spin.setValue(1)
        self.num_pages_spin.valueChanged.connect(self.init_trees)
        config_row.addWidget(num_pages_label)
        config_row.addWidget(self.num_pages_spin)

        config_label = QLabel("Slots per Page (Single number or list):")
        self.page_config_edit = QLineEdit()
        self.page_config_edit.setPlaceholderText("e.g. 4, 8, 4, 2")
        self.page_config_edit.editingFinished.connect(self.on_page_config_changed)
        config_row.addWidget(config_label)
        config_row.addWidget(self.page_config_edit)
        right_layout.addLayout(config_row)
        
        title_row = QHBoxLayout()
        title_label = QLabel("Page Title")
        self.title_edit = QLineEdit()
        self.title_edit.setText("default title")
        self.title_edit.returnPressed.connect(self.on_title_return_pressed)
        
        self.apply_all_title_btn = QPushButton("Apply to All")
        self.apply_all_title_btn.clicked.connect(self.apply_title_to_all)
        
        title_row.addWidget(title_label)
        title_row.addWidget(self.title_edit)
        title_row.addWidget(self.apply_all_title_btn)
        right_layout.addLayout(title_row)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Optimization Algorithm")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Simulated Annealing", "sa")
        self.mode_combo.addItem("Evolutionary Strategy", "es")
        self.mode_combo.addItem("Linear Partitioning (Instant)", "linear")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo)
        right_layout.addLayout(mode_row)

        steps_row = QHBoxLayout()
        steps_label = QLabel("Annealing Steps")
        self.steps_spin = QSpinBox()
        self.steps_spin.setMinimum(1000)
        self.steps_spin.setMaximum(100000)
        self.steps_spin.setValue(5000)
        self.steps_spin.setSingleStep(1000)
        steps_row.addWidget(steps_label)
        steps_row.addWidget(self.steps_spin)
        right_layout.addLayout(steps_row)

        # Labels and Gap
        label_gap_row = QHBoxLayout()
        self.show_labels_cb = QCheckBox("Show Filenames")
        self.show_labels_cb.setChecked(self.show_labels)
        self.show_labels_cb.toggled.connect(self.on_show_labels_toggled)
        
        self.show_page_numbers_cb = QCheckBox("Page Numbers")
        self.show_page_numbers_cb.setChecked(self.show_page_numbers)
        self.show_page_numbers_cb.toggled.connect(self.on_show_page_numbers_toggled)

        self.label_bold_cb = QCheckBox("Bold")
        self.label_bold_cb.setChecked(self.label_bold)
        self.label_bold_cb.toggled.connect(self.on_label_bold_toggled)

        label_size_label = QLabel("Size %")
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(10, 100)
        self.label_size_spin.setValue(int(self.label_size_ratio * 100))
        self.label_size_spin.valueChanged.connect(self.on_label_size_changed)
        
        gap_label = QLabel("Gap")
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 100)
        self.gap_spin.setValue(self.image_gap)
        self.gap_spin.valueChanged.connect(self.on_gap_changed)
        
        label_gap_row.addWidget(self.show_labels_cb)
        label_gap_row.addWidget(self.show_page_numbers_cb)
        label_gap_row.addWidget(self.label_bold_cb)
        label_gap_row.addStretch()
        label_gap_row.addWidget(label_size_label)
        label_gap_row.addWidget(self.label_size_spin)
        label_gap_row.addWidget(gap_label)
        label_gap_row.addWidget(self.gap_spin)
        right_layout.addLayout(label_gap_row)

        self.optimize_btn = QPushButton("✨ Optimize Layout")
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.optimize_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset Layout")
        self.reset_btn.clicked.connect(self.reset_layout)
        self.export_btn = QPushButton("Save as PDF")
        self.export_btn.clicked.connect(self.export_pdf_dialog)
        self.export_btn.setEnabled(False)

        crop_row = QHBoxLayout()
        self.crop_all_btn = QPushButton("Crop All")
        self.crop_all_btn.clicked.connect(self.crop_all)
        self.uncrop_all_btn = QPushButton("Uncrop All")
        self.uncrop_all_btn.clicked.connect(self.uncrop_all)
        crop_row.addWidget(self.crop_all_btn)
        crop_row.addWidget(self.uncrop_all_btn)
        
        self.progress_bar = QProgressBar()
        
        right_layout.addWidget(self.optimize_btn)
        right_layout.addWidget(self.reset_btn)
        right_layout.addWidget(self.export_btn)
        right_layout.addLayout(crop_row)
        right_layout.addStretch()

        # Plot Placeholder
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        right_layout.addWidget(self.plot_container)
        self.figure = None
        self.canvas = None
        self.ax = None

        right_layout.addWidget(self.progress_bar)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.view)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Bottom: Navigation
        bottom_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Page")
        self.prev_btn.clicked.connect(self.prev_page)
        self.page_label = QLabel("Page 1 / 1")
        self.next_btn = QPushButton("Next Page")
        self.next_btn.clicked.connect(self.next_page)
        
        self.grid_view_btn = QPushButton("Grid View")
        self.grid_view_btn.setCheckable(True)
        self.grid_view_btn.clicked.connect(self.toggle_view_mode)
        
        bottom_layout.addWidget(self.prev_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.page_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.grid_view_btn)
        bottom_layout.addWidget(self.next_btn)
        main_layout.addLayout(bottom_layout)

    def load_images_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images(folder)

    def take_snapshot(self):
        if not self.current_folder:
            return

        # Deep copy mutable structures
        perms_copy = copy.deepcopy(self.pages_perms)
        locks_copy = copy.deepcopy(self.pages_locks)
        crop_states_copy = self.image_crop_states.copy()
        # Convert set to list for JSON serialization
        forced_aspects_copy = list(self.forced_aspect_ratios)

        snapshot_data = {
            "num_pages": self.num_pages_spin.value(),
            "page_config": self.page_config_edit.text(),
            "page_titles": self.page_titles,
            "trees": [serialize_tree(root) for root in self.pages_roots],
            "perms": perms_copy,
            "locks": locks_copy,
            "crop_states": crop_states_copy,
            "forced_aspects": forced_aspects_copy,
            "settings": {
                "image_gap": self.image_gap,
                "show_labels": self.show_labels,
                "label_bold": self.label_bold,
                "label_size_ratio": self.label_size_ratio,
                "show_page_numbers": self.show_page_numbers
            }
        }
        
        # Find a unique filename
        base_name = self.current_folder.name
        i = 1
        while True:
            snapshot_path = self.current_folder / f"{base_name}_snapshot_{i}.json"
            if not snapshot_path.exists():
                break
            i += 1
            
        try:
            with open(snapshot_path, "w") as f:
                json.dump(snapshot_data, f, indent=4)
            
            self.snapshots.append({"path": snapshot_path, "data": snapshot_data})
            self.snapshot_combo.addItem(snapshot_path.name)
            self.snapshot_combo.setCurrentIndex(len(self.snapshots) - 1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save snapshot: {e}")

    def delete_snapshot(self):
        idx = self.snapshot_combo.currentIndex()
        if idx < 0 or idx >= len(self.snapshots):
            return
            
        snapshot = self.snapshots[idx]
        path = snapshot["path"]
        
        try:
            if path.exists():
                path.unlink()
            
            self.snapshots.pop(idx)
            self.snapshot_combo.removeItem(idx)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete snapshot: {e}")

    def restore_snapshot(self):
        idx = self.snapshot_combo.currentIndex()
        if idx < 0 or idx >= len(self.snapshots):
            return
            
        data = self.snapshots[idx]["data"]
        
        # Restore UI values
        self.num_pages_spin.setValue(data["num_pages"])
        self.page_config_edit.setText(data["page_config"])
        
        # Restore Settings
        settings = data.get("settings", {})
        self.image_gap = settings.get("image_gap", 10)
        self.show_labels = settings.get("show_labels", True)
        self.label_bold = settings.get("label_bold", False)
        self.label_size_ratio = settings.get("label_size_ratio", 0.5)
        self.show_page_numbers = settings.get("show_page_numbers", False)
        
        self.gap_spin.setValue(self.image_gap)
        self.show_labels_cb.setChecked(self.show_labels)
        self.label_bold_cb.setChecked(self.label_bold)
        self.label_size_spin.setValue(int(self.label_size_ratio * 100))
        self.show_page_numbers_cb.setChecked(self.show_page_numbers)

        # Restore Data
        self.page_titles = data.get("page_titles", [data.get("title", "default title")] * data["num_pages"])
        self.pages_roots = [deserialize_tree(t) for t in data["trees"]]
        self.pages_perms = copy.deepcopy(data["perms"])
        self.pages_locks = [{int(k): v for k, v in d.items()} for d in data["locks"]]
        self.image_crop_states = {int(k): v for k, v in data["crop_states"].items()}
        # Convert list back to set
        self.forced_aspect_ratios = set(data["forced_aspects"])

        self.current_page_idx = 0
        self.update_page_nav()
        self.update_stats()
        self.draw_layout()

    def load_images(self, folder):
        self.image_paths = []
        self.current_folder = Path(folder)
        self.page_titles = []
        self.image_metadata = []
        self.all_prefs = []
        self.pages_roots = []
        self.pages_perms = []
        self.pages_locks = []
        self.target_leaf_count = None
        self.image_list.clear()
        
        folder_path = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted([p for p in folder_path.glob("*") if p.suffix.lower() in exts])
        
        if not files:
            self.slot_combo.clear()
            self.slot_combo.setEnabled(False)
            self.target_leaf_count = None
            self.scene.clear()
            self.optimize_btn.setEnabled(False)
            return

        self.image_paths = files
        
        # Task 2: Image Metadata Caching
        self.image_metadata = sa.batch_process_images(self.image_paths)
        self.all_prefs = [m.pref_aspect for m in self.image_metadata]
        
        # Load existing snapshots
        self.snapshots = []
        self.snapshot_combo.clear()
        snapshot_files = sorted(list(self.current_folder.glob(f"{self.current_folder.name}_snapshot_*.json")))
        for p in snapshot_files:
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                self.snapshots.append({"path": p, "data": data})
                self.snapshot_combo.addItem(p.name)
            except Exception as e:
                print(f"Error loading snapshot {p}: {e}")

        # Default config: 1 page with power of 2 slots >= total images
        if self.image_paths:
            default_slots = sa.largest_power_of_two_leq(len(self.image_paths))
            if default_slots < len(self.image_paths):
                default_slots *= 2
            self.page_config_edit.setText(str(default_slots))
        
        for idx, p in enumerate(self.image_paths):
            item = QListWidgetItem(self.image_list)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setSizeHint(QSize(200, 70))
            
            pix = QPixmap(str(p)).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            widget = ImageListEntry(pix, p.name)
            # Connect pool toggle to stats update
            widget.pool_cb.toggled.connect(lambda: self.update_stats())
            widget.mandatory_cb.toggled.connect(lambda: self.update_stats())
            self.image_list.setItemWidget(item, widget)
            
        self.init_trees()
        self.update_stats()
        self.optimize_btn.setEnabled(bool(self.pages_roots))

    def update_stats(self):
        total_images = len(self.image_paths)
        
        # Calculate album capacity
        album_capacity = 0
        if self.pages_roots:
            for root in self.pages_roots:
                album_capacity += len(sa.leaf_ids(root))
        
        # Calculate mandatory and pool
        mandatory_count = 0
        pool_count = 0
        
        # Include locked images as mandatory
        locked_indices = set()
        for p_locks in self.pages_locks:
            for img_idx in p_locks.values():
                locked_indices.add(img_idx)
        
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                idx = item.data(Qt.ItemDataRole.UserRole)
                is_mandatory = widget.mandatory_cb.isChecked() or (idx in locked_indices)
                is_pool = widget.pool_cb.isChecked()
                
                if is_mandatory:
                    mandatory_count += 1
                elif is_pool:
                    pool_count += 1
                    
        self.stat_total_images.setText(f"Total Images: {total_images}")
        self.stat_album_capacity.setText(f"Album Capacity: {album_capacity}")
        self.stat_mandatory.setText(f"Mandatory: {mandatory_count}")
        self.stat_pool.setText(f"Pool: {pool_count}")

    def on_page_config_changed(self):
        self.init_trees()

    def init_trees(self):
        total_images = len(self.image_paths)
        if total_images == 0:
            self.optimize_btn.setEnabled(False)
            return

        config_str = self.page_config_edit.text().strip()
        page_counts = []
        
        try:
            if "," in config_str:
                # List Mode: Parse comma-separated values
                page_counts = [int(s.strip()) for s in config_str.split(",") if s.strip()]
                # Update spinbox to match list length
                self.num_pages_spin.blockSignals(True)
                self.num_pages_spin.setValue(len(page_counts))
                self.num_pages_spin.blockSignals(False)
            else:
                # Uniform Mode: Single value or empty
                if config_str:
                    slots_per_page = int(config_str)
                else:
                    slots_per_page = 4 # Default
                
                num_pages = self.num_pages_spin.value()
                page_counts = [slots_per_page] * num_pages
                
            # Validate powers of two
            for c in page_counts:
                if not sa.is_power_of_two(c):
                    raise ValueError(f"Invalid slot count: {c}. Must be power of 2.")
                    
        except ValueError as e:
            # Fallback or error
            print(f"Config error: {e}")
            if not self.pages_roots: # Only if we don't have a valid state
                 page_counts = [4]
            else:
                return

        if not page_counts:
             page_counts = [4]

        num_pages = len(page_counts)
        
        # Update page titles
        old_titles = self.page_titles
        self.page_titles = []
        default_title = self.current_folder.name if self.current_folder else "Page"
        for i in range(num_pages):
            if i < len(old_titles):
                self.page_titles.append(old_titles[i])
            else:
                self.page_titles.append(f"{default_title} {i+1}")
        
        # Update title edit if current page changed or initialized
        if self.current_page_idx >= num_pages:
            self.current_page_idx = num_pages - 1
        if self.current_page_idx < 0:
            self.current_page_idx = 0
            
        if self.page_titles:
            self.title_edit.blockSignals(True)
            self.title_edit.setText(self.page_titles[self.current_page_idx])
            self.title_edit.blockSignals(False)

        try:
            self.pages_roots = [sa.build_full_tree(count, seed=42 + i) for i, count in enumerate(page_counts)]
        except AssertionError:
            self.pages_roots = []
            self.optimize_btn.setEnabled(False)
            return

        # Re-distribute pool indices
        pool_indices = list(range(total_images))
        random.shuffle(pool_indices)
        
        self.pages_perms = []
        current_idx = 0
        for count in page_counts:
            # Take 'count' images from pool if available, else fill with -1 or wrap?
            # The original code sliced. If we run out of images, we might have issues.
            # But the optimization loop handles perms.
            # Let's just slice safely.
            end_idx = min(current_idx + count, total_images)
            page_perm = pool_indices[current_idx:end_idx]
            # If not enough images, fill with random ones or just leave it short?
            # The system expects perm length == leaf count usually?
            # sa_advanced.py: energy uses perm[lid]. So perm must map leaf_id -> img_idx.
            # If we don't have enough images, we must fill with something valid or handle it.
            # For now, let's cycle if needed or just fill with 0.
            while len(page_perm) < count:
                if total_images > 0:
                    page_perm.append(random.choice(pool_indices))
                else:
                    page_perm.append(0) # Should not happen if total_images > 0
            
            self.pages_perms.append(page_perm)
            current_idx += count

        # Reset locks if page count changed or structure changed significantly
        # Ideally we try to preserve locks if possible, but with variable pages it's hard to map.
        # For simplicity, if the number of pages changes, we might lose locks.
        # But let's try to keep them if index matches.
        if len(self.pages_locks) != num_pages:
            self.pages_locks = [{} for _ in range(num_pages)]
        else:
            # Clear locks that are out of bounds for new page sizes
            for i in range(num_pages):
                limit = page_counts[i]
                to_remove = [k for k in self.pages_locks[i] if k >= limit]
                for k in to_remove:
                    del self.pages_locks[i][k]
        
        self.current_page_idx = 0
        self.update_page_nav()
        self.update_stats()
        self.draw_layout()
        self.optimize_btn.setEnabled(bool(self.pages_roots))
        self.export_btn.setEnabled(bool(self.pages_roots))

    def draw_layout(self):
        self.scene.clear()
        if not self.pages_roots:
            return
        
        if self.view_mode == "single":
            if self.current_page_idx >= len(self.pages_roots):
                return
            self.render_page_to_scene(self.current_page_idx, 0, 0)
        else:
            # Grid View
            cols = 3
            visual_gap = 100
            for i in range(len(self.pages_roots)):
                row = i // cols
                col = i % cols
                x_offset = col * (self.page_W + visual_gap)
                y_offset = row * (self.page_H + visual_gap)
                self.render_page_to_scene(i, x_offset, y_offset)
                
                # Add page label
                label = QGraphicsTextItem(f"Page {i + 1}")
                font = label.font()
                font.setPixelSize(40)
                font.setBold(True)
                label.setFont(font)
                label.setDefaultTextColor(QColor(200, 200, 200))
                label.setPos(x_offset, y_offset - 60)
                self.scene.addItem(label)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def render_page_to_scene(self, page_idx, x_offset, y_offset):
        root = self.pages_roots[page_idx]
        perm = self.pages_perms[page_idx]
        locked = self.pages_locks[page_idx]
        
        margin = 20
        W, H = self.page_W, self.page_H
        title_height = int(H * 0.1)
        footer_height = int(H * 0.05) if self.show_page_numbers else 0
        in_W = W - 2*margin
        in_H = H - 2*margin - title_height - footer_height
        
        # Draw page background
        bg_rect = QGraphicsRectItem(x_offset, y_offset, W, H)
        bg_rect.setBrush(QColor(255, 255, 255))
        bg_rect.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(bg_rect)

        boxes = sa.decode_region(root, margin, margin + title_height, in_W, in_H)
        
        # Draw title
        title_text = self.page_titles[page_idx] if page_idx < len(self.page_titles) else "Page"
        text_item = QGraphicsTextItem()
        text_document = QTextDocument()
        text_document.setHtml(f"<div style='text-align: center;'>{title_text}</div>")
        text_item.setDocument(text_document)
        text_item.setPos(x_offset + margin, y_offset + margin)
        text_item.setTextWidth(W - 2*margin)
        font = text_item.font()
        font.setPixelSize(int(title_height * 0.4))
        text_item.setFont(font)
        text_item.setDefaultTextColor(QColor(0, 0, 0))
        self.scene.addItem(text_item)

        # Draw page number
        if self.show_page_numbers:
            page_num_text = str(page_idx + 1)
            num_item = QGraphicsTextItem()
            num_doc = QTextDocument()
            num_doc.setHtml(f"<div style='text-align: center;'>{page_num_text}</div>")
            num_item.setDocument(num_doc)
            num_item.setPos(x_offset + margin, y_offset + H - margin - footer_height)
            num_item.setTextWidth(W - 2*margin)
            n_font = num_item.font()
            n_font.setPixelSize(int(footer_height * 0.5))
            num_item.setFont(n_font)
            num_item.setDefaultTextColor(QColor(0, 0, 0))
            self.scene.addItem(num_item)
        
        gap = self.image_gap
        
        for leaf_id, (x, y, w, h) in boxes.items():
            # Apply gap
            rect_item = LeafItem(
                0, 0, w - gap, h - gap, 
                page_idx, leaf_id, self
            )
            rect_item.setPos(x_offset + x + gap/2, y_offset + y + gap/2)
            
            if leaf_id >= len(perm):
                continue
            img_idx = perm[leaf_id]
            if img_idx < 0 or img_idx >= len(self.image_paths):
                continue
            
            # Load image to display in rect
            path = self.image_paths[img_idx]
            pix = QPixmap(str(path))
            if not pix.isNull():
                 should_crop = self.image_crop_states.get(img_idx, True)
                 if should_crop:
                    scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    copy = scaled.copy(
                        (scaled.width() - int(w-gap)) // 2,
                        (scaled.height() - int(h-gap)) // 2,
                        int(w-gap), int(h-gap)
                    )
                    rect_item.pixmap_item.setPixmap(copy)
                 else:
                    scaled = pix.scaled(int(w-gap), int(h-gap), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    rect_item.pixmap_item.setPixmap(scaled)
                    rect_item.pixmap_item.setPos((w - gap - scaled.width()) / 2, (h - gap - scaled.height()) / 2)
            
            if self.show_labels:
                label_text = path.stem
                label_item = QGraphicsTextItem(label_text, rect_item)
                font = label_item.font()
                font_size = max(8, int(self.image_gap * self.label_size_ratio))
                font.setPixelSize(font_size)
                font.setBold(self.label_bold)
                label_item.setFont(font)
                label_item.setDefaultTextColor(Qt.GlobalColor.black)
                
                # Center horizontally
                l_w = label_item.boundingRect().width()
                l_h = label_item.boundingRect().height()
                label_item.setPos((w - gap - l_w) / 2, -l_h)

            if leaf_id in locked:
                rect_item.set_locked(True)
                
            self.scene.addItem(rect_item)

    def handle_drop(self, page_idx, leaf_id, img_idx):
        # User dropped image `img_idx` onto `leaf_id` of `page_idx`.
        # Constraint: Image `img_idx` MUST be at `leaf_id`.
        if not self.pages_roots or page_idx >= len(self.pages_roots) or not self.pages_perms[page_idx]:
            return
        perm = self.pages_perms[page_idx]
        locked = self.pages_locks[page_idx]
        if leaf_id < 0 or leaf_id >= len(perm):
            return
        if img_idx < 0 or img_idx >= len(self.image_paths):
            return

        # Remove any previous lock that tied this image to a different leaf on the SAME page.
        for locked_leaf, locked_img in list(locked.items()):
            if locked_leaf != leaf_id and locked_img == img_idx:
                del locked[locked_leaf]

        locked[leaf_id] = img_idx

        if img_idx in perm:
            current_leaf_of_img = perm.index(img_idx)
            if current_leaf_of_img != leaf_id:
                perm[leaf_id], perm[current_leaf_of_img] = (
                    perm[current_leaf_of_img],
                    perm[leaf_id],
                )
        else:
            perm[leaf_id] = img_idx

        # Automatically set crop state to False (Fit) for locked slots
        self.image_crop_states[img_idx] = False
        self.forced_aspect_ratios.add(img_idx)

        self.update_stats()
        self.draw_layout()

    def toggle_crop_state(self, img_idx):
        current = self.image_crop_states.get(img_idx, True)
        self.image_crop_states[img_idx] = not current
        self.draw_layout()

    def toggle_aspect_lock(self, img_idx):
        if img_idx in self.forced_aspect_ratios:
            self.forced_aspect_ratios.remove(img_idx)
        else:
            self.forced_aspect_ratios.add(img_idx)
            self.image_crop_states[img_idx] = False
        self.draw_layout()

    def on_show_labels_toggled(self, checked):
        self.show_labels = checked
        self.draw_layout()

    def on_label_bold_toggled(self, checked):
        self.label_bold = checked
        self.draw_layout()

    def on_label_size_changed(self, value):
        self.label_size_ratio = value / 100.0
        self.draw_layout()

    def on_gap_changed(self, value):
        self.image_gap = value
        self.draw_layout()

    def crop_all(self):
        for i in range(len(self.image_paths)):
            self.image_crop_states[i] = True
        self.draw_layout()

    def uncrop_all(self):
        for i in range(len(self.image_paths)):
            self.image_crop_states[i] = False
        self.draw_layout()

    def start_optimization(self):
        if not self.pages_roots or not self.pages_perms:
            return

        # Determine page configuration from current roots
        page_counts = [len(sa.leaf_ids(r)) for r in self.pages_roots]
        num_pages = len(page_counts)
        total_needed = sum(page_counts)

        # Check for existing locks to decide whether to rebuild trees
        has_locks = False
        for root in self.pages_roots:
            if root and any(n.locked for n in sa.internal_nodes(root)):
                has_locks = True
                break

        # Relaunch: Regenerate trees to avoid local minima, preserving locks
        if not has_locks:
            self.pages_roots = [sa.build_full_tree(count, seed=random.randint(0, 1000000)) for count in page_counts]

        # Step A: Identify locked indices (implicitly mandatory)
        locked_indices = set()
        for p_locks in self.pages_locks:
            for img_idx in p_locks.values():
                locked_indices.add(img_idx)

        # Step B: Gather mandatory and optional images from the list
        mandatory_indices = set(locked_indices)
        optional_indices = []

        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                idx = item.data(Qt.ItemDataRole.UserRole)
                if widget.mandatory_cb.isChecked():
                    mandatory_indices.add(idx)
                elif widget.pool_cb.isChecked() and idx not in locked_indices:
                    optional_indices.append(idx)

        # Step C: Validation
        if len(mandatory_indices) > total_needed:
            QMessageBox.warning(self, "Too Many Mandatory Images", 
                                f"You have {len(mandatory_indices)} mandatory images (including locked ones) but only {total_needed} slots available.")
            return

        if len(mandatory_indices) + len(optional_indices) < total_needed:
            QMessageBox.warning(self, "Not Enough Images", 
                                f"You need {total_needed} images, but only {len(mandatory_indices) + len(optional_indices)} are available in the pool.")
            return

        # Step D: Selection
        # Start with mandatory images
        active_indices = list(mandatory_indices)
        # Fill the rest with optional images (sequential selection)
        slots_remaining = total_needed - len(active_indices)
        if slots_remaining > 0:
            active_indices.extend(optional_indices[:slots_remaining])

        # Initialize buckets
        chunks = [[] for _ in range(num_pages)]
        assigned = set()

        # Identify and assign locked images to their specific pages
        for p in range(num_pages):
            for leaf_id, img_idx in self.pages_locks[p].items():
                if img_idx in active_indices:
                    chunks[p].append(img_idx)
                    assigned.add(img_idx)

        # Distribute remaining active images into chunks
        free_active = [idx for idx in active_indices if idx not in assigned]
        for p in range(num_pages):
            # Use specific slot count for this page
            slots_for_page = page_counts[p]
            while len(chunks[p]) < slots_for_page and free_active:
                chunks[p].append(free_active.pop(0))

        # Create swap pool from any remaining optional images not in active_indices
        swap_pool = optional_indices[slots_remaining:]

        # Prepare forced_aspects
        forced_aspects = {}
        for idx in self.forced_aspect_ratios:
            if 0 <= idx < len(self.image_metadata):
                forced_aspects[idx] = self.image_metadata[idx].aspect_ratio

        # Prepare for thread
        title = self.title_edit.text()
        steps = self.steps_spin.value()
        mode = self.mode_combo.currentData()
        self.worker = OptimizationThread(
            chunks, swap_pool, self.pages_locks, self.all_prefs, self.image_paths, self.page_W, self.page_H, title, self.pages_roots, steps, mode, forced_aspects, self.show_page_numbers
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_optim.connect(self.on_optim_finished)
        
        self.optimize_btn.setEnabled(False)
        self.worker.start()

    def on_optim_finished(self, results, energy_history):
        self.pages_perms = results
        self.update_page_nav()
        self.draw_layout()
        self.optimize_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.update_energy_plot(energy_history)

    def update_energy_plot(self, history):
        if self.canvas is None:
            self.figure = Figure(figsize=(5, 3))
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax.clear()

        self.ax.plot(history)
        self.ax.set_title("Energy Curve")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Energy")
        self.figure.tight_layout()
        self.canvas.draw()

    def export_pdf_dialog(self):
        if not self.current_folder:
            return
            
        pdf_dir = self.current_folder
        
        base_name = self.current_folder.name
        i = 1
        while True:
            path = pdf_dir / f"{base_name}_{i}.pdf"
            if not path.exists():
                break
            i += 1
            
        self.export_btn.setEnabled(False)
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.export_worker = ExportThread(
            str(path), self.pages_roots, self.pages_perms, self.image_paths, self.page_titles, 
            self.image_crop_states, self.show_labels, self.image_gap, self.label_bold, self.label_size_ratio,
            self.show_page_numbers
        )
        self.export_worker.progress.connect(self.progress_bar.setValue)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.start()

    def on_export_finished(self, success, message):
        self.export_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        if not success:
            QMessageBox.critical(self, "Export Failed", message)

    def reset_layout(self):
        for locks in self.pages_locks:
            locks.clear()
        self.init_trees() # Re-randomize

    def toggle_view_mode(self):
        if self.grid_view_btn.isChecked():
            self.view_mode = "grid"
            self.grid_view_btn.setText("Single View")
        else:
            self.view_mode = "single"
            self.grid_view_btn.setText("Grid View")
        self.update_page_nav()
        self.draw_layout()

    def prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.update_page_nav()
            self.draw_layout()

    def next_page(self):
        if self.current_page_idx < len(self.pages_roots) - 1:
            self.current_page_idx += 1
            self.update_page_nav()
            self.draw_layout()

    def update_page_nav(self):
        num_pages = len(self.pages_roots)
        if num_pages == 0:
            self.page_label.setText("Page 0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        elif self.view_mode == "grid":
            self.page_label.setText(f"All Pages ({num_pages})")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.page_label.setText(f"Page {self.current_page_idx + 1} / {num_pages}")
            self.prev_btn.setEnabled(self.current_page_idx > 0)
            self.next_btn.setEnabled(self.current_page_idx < num_pages - 1)
            
            # Update title edit for current page
            if 0 <= self.current_page_idx < len(self.page_titles):
                self.title_edit.blockSignals(True)
                self.title_edit.setText(self.page_titles[self.current_page_idx])
                self.title_edit.blockSignals(False)

    def on_title_return_pressed(self):
        if 0 <= self.current_page_idx < len(self.page_titles):
            self.page_titles[self.current_page_idx] = self.title_edit.text()
            self.draw_layout()

    def apply_title_to_all(self):
        text = self.title_edit.text()
        for i in range(len(self.page_titles)):
            self.page_titles[i] = text
        self.draw_layout()

    def on_show_page_numbers_toggled(self, checked):
        self.show_page_numbers = checked
        self.draw_layout()

    def select_all_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                widget.pool_cb.setChecked(True)

    def select_none_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if isinstance(widget, ImageListEntry):
                widget.pool_cb.setChecked(False)

# Custom Drag for ListWidget
# We need to subclass QListWidget to support custom mime data easily, 
# or just rely on internal model?
# Actually, standard QListWidget drag puts mime data.
# Often easier to override startDrag in QListWidget
# But let's try to patch QListWidget instance methods or subclass properly.

# Monkey-patching for simplicity in one file
def startDrag(self, actions):
    drag = QDrag(self)
    items = self.selectedItems()
    if not items: return
    item = items[0]
    idx = item.data(Qt.ItemDataRole.UserRole)
    
    mime = QMimeData()
    mime.setData("application/x-image-idx", str(idx).encode())
    drag.setMimeData(mime)
    
    drag.exec(Qt.DropAction.CopyAction)

QListWidget.startDrag = startDrag


def main():
    app = QApplication(sys.argv)
    window = AlbumWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
